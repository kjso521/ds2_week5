import glob
import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Literal
from pathlib import Path
import re
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import itertools

from .noise_simulator import NoiseSimulator, NoisyType
from dataset.forward_simulator import ForwardSimulator


prob_flip: float = 0.5


class DataKey(IntEnum):
    image_gt = 0
    image_noise = 1
    name = 2


@dataclass
class LoaderConfig:
    # --- 재설계: 실시간 증강을 위한 설정들로 변경 ---
    data_type: str
    batch: int
    num_workers: int
    shuffle: bool
    augmentation_mode: Literal['none', 'noise_only', 'conv_only', 'both']
    noise_type: Literal["gaussian"]
    noise_levels: list[float]
    conv_directions: list[tuple[float, float]]


class RandomDataWrapper(Dataset):
    file_list: list[str]
    training_mode: bool
    
    # --- 재설계: 실시간 증강을 위한 멤버 변수 ---
    augmentation_mode: Literal['none', 'noise_only', 'conv_only', 'both']
    noise_levels: list[float]
    conv_directions: list[tuple[float, float]]
    noise_type: NoisyType
    noise_simulator: NoiseSimulator
    forward_simulator: ForwardSimulator

    def __init__(
        self,
        file_path: list[str],
        data_type: str,
        training_mode: bool,
        # --- 재설계: 실시간 증강 파라미터를 직접 받음 ---
        augmentation_mode: Literal['none', 'noise_only', 'conv_only', 'both'],
        noise_type: NoisyType,
        noise_levels: list[float],
        conv_directions: list[tuple[float, float]],
    ):
        super().__init__()
        self.training_mode = training_mode
        self.augmentation_mode = augmentation_mode
        self.noise_type = noise_type
        self.noise_levels = noise_levels
        self.conv_directions = conv_directions

        # 시뮬레이터들을 미리 초기화해둡니다.
        self.noise_simulator = NoiseSimulator(noise_type=self.noise_type, noise_sigma=0.0) # sigma는 __getitem__에서 매번 덮어씀
        self.forward_simulator = ForwardSimulator()

        total_list: list[str] = []
        for _file_path in file_path:
            p = Path(_file_path)
            total_list += [str(f) for f in p.glob(data_type)]
        self.file_list = sorted(total_list)


    @staticmethod
    def _load_from_npy(
        file_npy: str,
    ) -> np.ndarray:
        img = np.load(file_npy).astype(np.float32)
        return img

    def _augment(
        self,
        img_np: np.ndarray,
    ) -> np.ndarray:
        if random.random() > prob_flip:
            img_np = np.ascontiguousarray(np.flip(img_np, axis=0))
        if random.random() > prob_flip:
            img_np = np.ascontiguousarray(np.flip(img_np, axis=1))
        return img_np

    def __getitem__(
        self,
        idx: int,
    ):
        image_gt_np = self._load_from_npy(self.file_list[idx])
        _name = Path(self.file_list[idx]).name

        if self.training_mode:
            image_gt_np = self._augment(image_gt_np)
        
        image_noise_np = image_gt_np.copy()
        
        if self.augmentation_mode == 'conv_only' or self.augmentation_mode == 'both':
            direction = random.choice(self.conv_directions)
            image_noise_np = self.forward_simulator.forward(image_noise_np, B0_dir=direction)
            
        if self.augmentation_mode == 'noise_only' or self.augmentation_mode == 'both':
            sigma = random.choice(self.noise_levels)
            image_noise_np = self.noise_simulator.add_noise(image_noise_np, sigma, self.noise_type)
        
        return {
            DataKey.image_gt: torch.from_numpy(image_gt_np.copy()).unsqueeze(0),
            DataKey.image_noise: torch.from_numpy(image_noise_np.copy()).unsqueeze(0),
            DataKey.name: _name,
        }

    def __len__(self) -> int:
        return len(self.file_list)

class ControlledDataWrapper(RandomDataWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_epoch = 0
        self.noise_conv_combinations = list(itertools.product(self.noise_levels, self.conv_directions))
        self.total_combinations = len(self.noise_conv_combinations)

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def __getitem__(self, idx: int):
        image_gt_np = self._load_from_npy(self.file_list[idx])
        _name = Path(self.file_list[idx]).name

        if self.training_mode:
            image_gt_np = self._augment(image_gt_np)

        image_noise_np = image_gt_np.copy()

        if self.augmentation_mode == 'noise_only':
            if len(self.noise_levels) > 0:
                noise_level = self.noise_levels[(self.current_epoch + idx) % len(self.noise_levels)]
                image_noise_np = self.noise_simulator.add_noise(image_noise_np, noise_level, self.noise_type)
        elif self.augmentation_mode == 'conv_only':
            if len(self.conv_directions) > 0:
                conv_direction = self.conv_directions[(self.current_epoch + idx) % len(self.conv_directions)]
                image_noise_np = self.forward_simulator.forward(image_noise_np, conv_direction)
        elif self.augmentation_mode == 'both':
            if self.total_combinations > 0:
                combination_idx = (self.current_epoch + idx) % self.total_combinations
                noise_level, conv_direction = self.noise_conv_combinations[combination_idx]
                
                image_noise_np = self.forward_simulator.forward(image_noise_np, conv_direction)
                image_noise_np = self.noise_simulator.add_noise(image_noise_np, noise_level, self.noise_type)

        return {
            DataKey.image_gt: torch.from_numpy(image_gt_np.copy()).unsqueeze(0),
            DataKey.image_noise: torch.from_numpy(image_noise_np.copy()).unsqueeze(0),
            DataKey.name: _name,
        }


def get_data_wrapper_loader(
    file_path: list[str],
    loader_cfg: LoaderConfig,
    training_mode: bool,
    data_wrapper_class: Literal['random', 'controlled'] = 'random',
) -> tuple[
    DataLoader,
    Dataset,
]:
    wrapper_map = {
        'random': RandomDataWrapper,
        'controlled': ControlledDataWrapper,
    }
    DataWrapperClass = wrapper_map[data_wrapper_class]
    
    dataset = DataWrapperClass(
        file_path=file_path,
        data_type=loader_cfg.data_type,
        training_mode=training_mode,
        augmentation_mode=loader_cfg.augmentation_mode if training_mode else 'none',
        noise_type=NoisyType.from_string(loader_cfg.noise_type),
        noise_levels=loader_cfg.noise_levels,
        conv_directions=loader_cfg.conv_directions,
    )
    
    if not len(dataset):
        return (None, None)

    dataloader = DataLoader(
        dataset,
        batch_size=loader_cfg.batch,
        num_workers=loader_cfg.num_workers,
        pin_memory=True,
        persistent_workers=loader_cfg.num_workers > 0,
        shuffle=loader_cfg.shuffle,
    )

    return (
        dataloader,
        dataset,
    )
