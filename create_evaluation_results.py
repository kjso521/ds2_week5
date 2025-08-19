"""
최종 평가를 위한 결과물(.npy 파일) 생성 스크립트.

이 스크립트는 학습된 DnCNN 모델(.ckpt)을 불러와 `test_y` 데이터셋의 모든 이미지를 복원하고,
`evaluate.ipynb`가 요구하는 형식에 맞춰 지정된 폴더에 .npy 파일로 저장합니다.
"""

import argparse
import os
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from code_denoising.common.logger import logger
from code_denoising.common.metric import calculate_psnr, calculate_ssim
from code_denoising.common.utils import (
    separator,
    validate_tensor_dimensions,
    validate_tensors,
)
from code_denoising.components.metriccontroller import MetricController
from code_denoising.datawrapper.datawrapper import DataKey, LoaderConfig, get_data_wrapper_loader
from code_denoising.model.dncnn import DnCNN
from code_denoising.params import DnCNNConfig

warnings.filterwarnings("ignore")

def create_results(
    data_loader: DataLoader,
    network: DnCNN,
    result_dir: Path,
    device: torch.device,
):
    """모델을 사용해 데이터셋을 복원하고 결과를 저장"""
    network.eval()
    test_state = MetricController()

    logger.info("Starting result generation...")
    for _data in tqdm(data_loader, desc="Processing"):
        noisy: Tensor = _data[DataKey.Noisy].to(device)
        label: Tensor = _data[DataKey.Label].to(device)
        names: list[str] = _data[DataKey.Name]

        validate_tensors([noisy, label])
        validate_tensor_dimensions([noisy, label], 4)

        with torch.no_grad():
            output = network(noisy)

        validate_tensors([output])
        validate_tensor_dimensions([output], 4)

        for idx in range(output.shape[0]):
            psnr = calculate_psnr(output[idx : idx + 1, ...], label[idx : idx + 1, ...])
            ssim = calculate_ssim(output[idx : idx + 1, ...], label[idx : idx + 1, ...])
            test_state.add("psnr", psnr)
            test_state.add("ssim", ssim)

            output_np = output[idx, 0, ...].cpu().numpy()
            file_name = names[idx]
            np.save(result_dir / file_name, output_np)

    logger.info(separator())
    logger.info("Result generation finished.")
    logger.info(f"Average PSNR: {test_state.mean('psnr'):.4f} (+/- {test_state.std('psnr'):.4f})")
    logger.info(f"Average SSIM: {test_state.mean('ssim'):.4f} (+/- {test_state.std('ssim'):.4f})")


def main():
    parser = argparse.ArgumentParser(description="Create result files for final evaluation.")
    parser.add_argument("--trained_checkpoints", type=str, required=True, help="Path to the trained model checkpoint (.ckpt).")
    parser.add_argument("--result_dir", type=str, default="result", help="Directory to save the resulting .npy files.")
    parser.add_argument("--data_root", type=str, default="dataset", help="Root directory of the dataset.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for processing.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use.")
    args = parser.parse_args()

    # --- 환경 설정 ---
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.name == 'nt' and args.num_workers > 0:
        logger.warning(f"Running on Windows. Forcing num_workers to 0 (was {args.num_workers}).")
        args.num_workers = 0

    result_path = Path(args.result_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Using device: {device}")
    logger.info(f"Results will be saved in: {result_path.resolve()}")

    # --- 모델 불러오기 ---
    logger.info(f"Loading checkpoint from: {args.trained_checkpoints}")
    checkpoint_data = torch.load(args.trained_checkpoints, map_location="cpu")
    model_config = DnCNNConfig(**checkpoint_data["model_config"])
    network = DnCNN(
        channels=model_config.channels,
        num_of_layers=model_config.num_of_layers,
        kernel_size=model_config.kernel_size,
        padding=model_config.padding,
        features=model_config.features,
    )
    
    load_state_dict = checkpoint_data["model_state_dict"]
    _state_dict = {k.replace("module.", ""): v for k, v in load_state_dict.items()}
    network.load_state_dict(_state_dict, strict=True)
    network = network.to(device)
    
    if torch.cuda.device_count() > 1:
        logger.info(f"Found {torch.cuda.device_count()} GPUs. Using DataParallel.")
        network = torch.nn.DataParallel(network)

    # --- 데이터 로더 설정 ---
    test_dataset_path = [str(Path(args.data_root) / "test_y")]
    loader_cfg = LoaderConfig(
        data_type="*.npy",
        batch=args.batch,
        num_workers=args.num_workers,
        shuffle=False,
        augmentation_mode='none',
        noise_type="gaussian",
        noise_levels=[],
        conv_directions=[],
    )
    data_loader, _, data_len = get_data_wrapper_loader(
        file_path=test_dataset_path,
        training_mode=False,
        loader_cfg=loader_cfg,
    )
    if not data_len:
        raise ValueError("Test dataset is empty.")

    # --- 결과 생성 실행 ---
    create_results(data_loader, network, result_path, device)


if __name__ == "__main__":
    main()
