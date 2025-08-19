"""
#  Copyright Juhyung Park
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : jack0878@snu.ac.kr
#
"""

import argparse
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch

# --- 경로 설정 로직 수정 ---
# Colab 환경인지 확인합니다 ('COLAB_GPU'는 Colab에서 기본으로 설정되는 환경 변수).
if 'COLAB_GPU' in os.environ:
    # Colab 환경이라면 로컬 런타임의 데이터셋 경로를 사용합니다.
    default_root: str = "/content/dataset"
else:
    # 로컬 환경이라면 프로젝트 상대 경로를 사용합니다.
    default_root: str = "dataset"

DATA_ROOT: str = os.environ.get("DATA_ROOT", default_root)

# 이제 학습 데이터셋은 항상 원본 'train' 폴더를 가리킵니다.
TRAIN_DATASET: list[str] = [
    DATA_ROOT + "/train",
]
VALID_DATASET: list[str] = [
    DATA_ROOT + "/val",
]
TEST_DATASET: list[str] = [
    DATA_ROOT + "/val", # test.py에서 평가용으로 사용
]

default_run_dir: str = "logs_dncnn"
RUN_DIR: str = os.environ.get("RUN_DIR", default_run_dir)

# --- 실시간 증강을 위한 파라미터 ---
# v1: 직접 분석한 노이즈 레벨
NOISE_LEVELS_V1 = [0.070, 0.132]
# v2: 팀원이 분석한 노이즈 레벨
NOISE_LEVELS_V2 = [0.0781, 0.1454]
# 추출된 컨볼루션 방향
CONV_DIRECTIONS = [
    (-0.8090, -0.5878),
    (-0.8090, 0.5878),
    (0.3090, -0.9511),
    (0.3090, 0.9511),
    (1.0000, 0.0000)
]

@dataclass
class GeneralConfig:
    # Dataset
    train_dataset: list[str] = field(default_factory=lambda: TRAIN_DATASET)
    valid_dataset: list[str] = field(default_factory=lambda: VALID_DATASET)
    test_dataset: list[str] = field(default_factory=lambda: TEST_DATASET)
    data_type: str = "*.npy"

    # Logging
    log_lv: str = "INFO"
    run_dir: Path = Path(RUN_DIR)
    init_time: float = 0.0

    # On-the-fly Augmentation
    augmentation_mode: Literal['none', 'noise_only', 'conv_only', 'both'] = 'both'
    noise_levels: list[float] = field(default_factory=lambda: NOISE_LEVELS_V1)
    conv_directions: list[tuple[float, float]] = field(default_factory=lambda: CONV_DIRECTIONS)

    # Model experiment
    model_type: Literal["dncnn", "unet"] = "dncnn"

    # Optimizer
    optimizer: Literal["adam", "adamw"] = "adam"
    loss_model: Literal["l1", "l2"] = "l2"
    lr: float = 1e-4
    lr_decay: float = 0.88
    lr_tol: int = 1

    # Train params
    gpu: str = "0"
    train_batch: int = 2
    valid_batch: int = 8
    train_epoch: int = 100
    logging_density: int = 4
    valid_interval: int = 2
    valid_tol: int = 2
    num_workers: int = 4
    save_val: bool = True
    parallel: bool = True
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_max_idx: int = 500

    # Experiment
    noise_type: Literal["gaussian"] = "gaussian" # 실시간 증강은 가우시안만 지원

    tag: str = ""


@dataclass
class DnCNNConfig:
    # Model architecture
    channels: int = 1
    num_of_layers: int = 17
    kernel_size: int = 3
    padding: int = 1
    features: int = 64


@dataclass
class UnetConfig:
    # Model architecture
    in_chans: int = 1
    out_chans: int = 1
    chans: int = 32
    num_pool_layers: int = 4


@dataclass
class TestConfig:
    # Dataset
    trained_checkpoints: str = ""


config = GeneralConfig()
dncnnconfig = DnCNNConfig()
unetconfig = UnetConfig()

# --- 아래의 모든 ArgumentParser 관련 로직을 if __name__ == "__main__": 블록으로 이동 ---
if __name__ == "__main__":
    # Argparser
    parser = argparse.ArgumentParser(description="Training Configuration")
    parser.add_argument("--run_dir_name", type=str, default=None, help="Override for the run directory name (e.g., logs_denoising)")
    general_config_dict = asdict(GeneralConfig())
    dncnn_config_dict = asdict(DnCNNConfig())
    unet_config_dict = asdict(UnetConfig())
    test_config_dict = asdict(TestConfig())

    for key, default_value in general_config_dict.items():
        if isinstance(default_value, bool):
            parser.add_argument(
                f"--{key}",
                type=lambda x: x.lower() in ("true", "t", "yes", "y", "1"),
                default=None,
                help=f"Set {key} (true/false, default: {default_value})",
            )
        else:
            parser.add_argument(
                f"--{key}",
                type=type(default_value),
                default=None,
                help=f"Override for {key}",
            )

    for key, default_value in dncnn_config_dict.items():
        if isinstance(default_value, bool):
            parser.add_argument(
                f"--{key}",
                type=lambda x: x.lower() in ("true", "t", "yes", "y", "1"),
                default=None,
                help=f"Set {key} (true/false, default: {default_value})",
            )
        else:
            parser.add_argument(
                f"--{key}",
                type=type(default_value),
                default=None,
                help=f"Override for {key}",
            )

    for key, default_value in unet_config_dict.items():
        if isinstance(default_value, bool):
            parser.add_argument(
                f"--{key}",
                type=lambda x: x.lower() in ("true", "t", "yes", "y", "1"),
                default=None,
                help=f"Set {key} (true/false, default: {default_value})",
            )
        else:
            parser.add_argument(
                f"--{key}",
                type=type(default_value),
                default=None,
                help=f"Override for {key}",
            )

    for key, default_value in test_config_dict.items():
        if isinstance(default_value, bool):
            parser.add_argument(
                f"--{key}",
                type=lambda x: x.lower() in ("true", "t", "yes", "y", "1"),
                default=None,
                help=f"Set {key} (true/false, default: {default_value})",
            )
        else:
            parser.add_argument(
                f"--{key}",
                type=type(default_value),
                default=None,
                help=f"Override for {key}",
            )

    # Apply argparser
    args = parser.parse_args()

    # 특별 인자들 먼저 처리
    if args.run_dir_name:
        config.run_dir = Path(args.run_dir_name)

    # 나머지 인자들은 루프를 통해 처리
    for key, value in vars(args).items():
        if value is not None:
            if hasattr(config, key):
                # bool 타입은 위에서 별도로 처리했으므로 여기서는 bool이 아닌 경우만 처리
                if not isinstance(getattr(config, key), bool):
                    setattr(config, key, value)

            if hasattr(dncnnconfig, key):
                if not isinstance(getattr(dncnnconfig, key), bool):
                    setattr(dncnnconfig, key, value)
            
            if hasattr(unetconfig, key):
                if not isinstance(getattr(unetconfig, key), bool):
                    setattr(unetconfig, key, value)

    # OS에 따른 num_workers 자동 조정 (Windows에서는 0으로 설정)
    if os.name == 'nt' and config.num_workers > 0:
        print(f"[INFO] Running on Windows. Forcing num_workers to 0 (was {config.num_workers}).")
        config.num_workers = 0

    # 사용 가능한 GPU가 1개 이하일 경우 DataParallel 기능 비활성화
    if torch.cuda.device_count() <= 1 and config.parallel:
        print(f"[INFO] Found {torch.cuda.device_count()} GPUs. Disabling DataParallel.")
        config.parallel = False

    # 이 스크립트를 직접 실행했을 때, 파싱된 config 내용을 출력하여 확인
    print("--- Parsed General Config ---")
    print(config)
    print("--- Parsed DnCNN Config ---")
    print(dncnnconfig)
    print("--- Parsed U-Net Config ---")
    print(unetconfig)
