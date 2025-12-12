"""
Trainer - 전략 독립적인 학습 엔진

각 전략(strategy_A~E)에서 설정을 전달받아 학습을 수행합니다.
증강, 손실함수, 옵티마이저, 스케줄러 등 모든 설정은 외부에서 주입됩니다.
AMP(Automatic Mixed Precision)를 지원하여 환경에 맞게 최적화합니다.

Example:
    from src import Trainer

    trainer = Trainer(
        model=model,
        criterion=nn.SmoothL1Loss(),
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        clip_grad=1.0,
        use_amp=True,  # auto_profile.py 환경변수 기반
    )

    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader, augment_fn=apply_mixup)
        val_loss = trainer.validate(val_loader)
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """
    전략 독립적 학습 엔진 (AMP 지원)

    Attributes:
        model: 학습할 모델
        criterion: 손실 함수
        optimizer: 옵티마이저
        device: 학습 디바이스 (cuda/cpu)
        scheduler: 학습률 스케줄러 (optional)
        clip_grad: 그래디언트 클리핑 값 (optional)
        use_amp: Automatic Mixed Precision 사용 여부
        amp_dtype: AMP 데이터 타입 ('fp16' 또는 'bf16')
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        clip_grad: Optional[float] = None,
        use_amp: bool = False,
        amp_dtype: str = "fp16",
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.clip_grad = clip_grad
        self.use_amp = use_amp and str(device).startswith("cuda")
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16

        # GradScaler for AMP
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

    def _get_autocast(self):
        """AMP autocast 컨텍스트 반환"""
        if self.use_amp:
            return torch.amp.autocast("cuda", dtype=self.amp_dtype)
        return nullcontext()

    def train_epoch(
        self,
        loader: DataLoader,
        augment_fn: Optional[Callable] = None,
    ) -> float:
        """
        한 에폭 학습 수행 (AMP 지원)

        Args:
            loader: 학습 데이터 로더
            augment_fn: 증강 함수 (optional)
                - None: 증강 없음
                - Callable[[x, y], [x_aug, y_aug]]: 배치 단위 증강

        Returns:
            평균 학습 손실
        """
        self.model.train()
        total_loss = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)

            # 증강 적용
            if augment_fn is not None:
                xb, yb = augment_fn(xb, yb)

            # Forward with AMP
            self.optimizer.zero_grad()
            with self._get_autocast():
                out = self.model(xb)
                loss = self.criterion(out, yb)

            # Backward with AMP
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.clip_grad is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()

            total_loss += loss.item()

        # 스케줄러 스텝 (에폭 기반)
        if self.scheduler is not None:
            self.scheduler.step()

        return total_loss / len(loader)

    def train_epoch_multi_aug(
        self,
        loader: DataLoader,
        augment_fns: list[Callable],
    ) -> float:
        """
        복수 증강을 적용한 학습 (aug_all 전략용, AMP 지원)

        원본 + 각 증강 결과를 concat하여 3배 배치로 학습

        Args:
            loader: 학습 데이터 로더
            augment_fns: 증강 함수 리스트 (예: [apply_mixup, apply_cutmix])

        Returns:
            평균 학습 손실
        """
        self.model.train()
        total_loss = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)

            # 원본 + 각 증강 결과 concat
            x_list = [xb]
            y_list = [yb]
            for aug_fn in augment_fns:
                x_aug, y_aug = aug_fn(xb, yb)
                x_list.append(x_aug)
                y_list.append(y_aug)

            xb_all = torch.cat(x_list, dim=0)
            yb_all = torch.cat(y_list, dim=0)

            # Forward with AMP
            self.optimizer.zero_grad()
            with self._get_autocast():
                out = self.model(xb_all)
                loss = self.criterion(out, yb_all)

            # Backward with AMP
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.clip_grad is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()

            total_loss += loss.item()

        if self.scheduler is not None:
            self.scheduler.step()

        return total_loss / len(loader)

    def validate(self, loader: DataLoader) -> float:
        """
        검증 수행 (AMP 지원)

        Args:
            loader: 검증 데이터 로더

        Returns:
            평균 검증 손실
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                with self._get_autocast():
                    out = self.model(xb)
                    total_loss += self.criterion(out, yb).item()

        return total_loss / len(loader)

    def save_checkpoint(self, path: str):
        """모델 체크포인트 저장"""
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path: str):
        """모델 체크포인트 로드"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
