"""
#  Copyright Juhyung Park
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : jack0878@snu.ac.kr
#
"""

import os
import sys
from pathlib import Path

# --- 중요: 모든 import 이전에 프로젝트 루트 경로를 시스템 경로에 추가 ---
# 이 스크립트가 실행되는 위치를 기준으로, 상위 2단계 폴더(week5)를 경로에 추가합니다.
# 이렇게 하면 'dataset' 폴더를 항상 찾을 수 있습니다.
sys.path.append(str(Path(__file__).resolve().parents[1]))


import time
import warnings
from dataclasses import asdict
from enum import Enum

import torch
from torch.nn import DataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.logger import logger, logger_add_handler
from common.utils import (
    call_next_id,
    separator,
)
from common.wrapper import error_wrap
from core_funcs import (
    get_loss_model,
    get_model,
    get_optimizer,
    save_checkpoint,
)
from datawrapper.datawrapper import DataKey, get_data_wrapper_loader, LoaderConfig
from params import config, dncnnconfig, unetconfig, parse_args_for_train_script

warnings.filterwarnings("ignore")


class Trainer:
    """Trainer"""

    def __init__(self) -> None:
        """__init__"""
        self.run_dir = Path(config.run_dir) / f"{call_next_id(Path(config.run_dir)):05d}_{config.tag or 'train'}"
        logger_add_handler(logger, f"{self.run_dir / 'log.log'}", config.log_lv)
        logger.info(separator())
        logger.info(f"Run dir: {self.run_dir}")
        os.makedirs(self.run_dir, exist_ok=True)

        logger.info(separator())
        logger.info("General Config")
        config_dict = asdict(config)
        for k in config_dict:
            logger.info(f"{k}:{config_dict[k]}")
        logger.info(separator())
        logger.info("Model Config")
        if config.model_type == "dncnn":
            model_config_dict = asdict(dncnnconfig)
        elif config.model_type == "unet":
            model_config_dict = asdict(unetconfig)
        else:
            model_config_dict = {}
        
        for k in model_config_dict:
            logger.info(f"{k}:{model_config_dict[k]}")

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        config.init_time = time.time()
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)

        self.best_metric: float = 0.0
        self.best_epoch: int = 0
        self.tol_count: int = 0
        self.global_step: int = 0
        self.primary_metric: float = 0.0

    def __call__(self) -> None:
        self._set_data()
        self._set_network()
        self._train()
        self._test(self.best_epoch, "best") # Test with the best model

    @error_wrap
    def _set_data(self) -> None:
        loader_cfg = LoaderConfig(
            data_type=config.data_type,
            batch=config.train_batch,
            num_workers=config.num_workers,
            shuffle=True,
            augmentation_mode=config.augmentation_mode,
            noise_type=config.noise_type,
            noise_levels=config.noise_levels,
            conv_directions=config.conv_directions
        )

        self.train_loader, self.train_dataset_obj = get_data_wrapper_loader(
            file_path=config.train_dataset,
            loader_cfg=loader_cfg,
            training_mode=True,
            data_wrapper_class='controlled'
        )
        logger.info(f"Train dataset length : {len(self.train_loader.dataset)}")

        # Update loader_cfg for validation/test
        loader_cfg.batch = config.valid_batch
        loader_cfg.shuffle = False

        self.valid_loader, _ = get_data_wrapper_loader(
            file_path=config.valid_dataset,
            loader_cfg=loader_cfg,
            training_mode=False,
            data_wrapper_class='controlled'
        )
        logger.info(f"Valid dataset length : {len(self.valid_loader.dataset)}")

        self.test_loader, _ = get_data_wrapper_loader(
            file_path=config.test_dataset,
            loader_cfg=loader_cfg,
            training_mode=False,
            data_wrapper_class='controlled'
        )
        logger.info(f"Test dataset length : {len(self.test_loader.dataset)}")

    @error_wrap
    def _set_network(self) -> None:
        self.model = get_model(config).to(config.device)
        self.optimizer = get_optimizer(config, self.model.parameters())
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', factor=config.lr_decay, patience=config.lr_tol)
        self.loss_model = get_loss_model(config).to(config.device)

        if config.parallel:
            self.model = DataParallel(self.model)

    @error_wrap
    def _train(self) -> None:
        logger.info(separator())
        logger.info("Train start")

        for epoch in range(config.train_epoch):
            logger.info(f"Epoch: {epoch}")
            logger.info(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.3e}")
            self.global_step = 0
            self.primary_metric = 0.0

            # Set the epoch for the controlled data loader
            if hasattr(self.train_dataset_obj, 'set_epoch'):
                self.train_dataset_obj.set_epoch(epoch)

            self.model.train()
            for i, data in enumerate(tqdm(self.train_loader, leave=False)):
                self.global_step += 1
                image_noise = data[DataKey.image_noise].to(config.device)
                image_gt = data[DataKey.image_gt].to(config.device)

                # Model prediction
                image_pred = self.model(image_noise)

                # Loss calculation
                self.optimizer.zero_grad()
                total_loss = self.loss_model(image_pred, image_gt)

                # Loss backward
                total_loss.backward()
                self.optimizer.step()

            if epoch % config.valid_interval == 0:
                is_best = self._valid()
                if is_best:
                    self.best_metric = self.primary_metric
                    self.best_epoch = epoch
                    self.tol_count = 0
                else:
                    self.tol_count += 1
                
                # Step the scheduler based on validation metric
                self.scheduler.step(self.primary_metric)

            if self.tol_count > config.valid_tol:
                logger.info("Early stopping triggered")
                break

    @error_wrap
    def _valid(self) -> bool:
        """Validation"""
        logger.info("Valid")
        primary_metric = test_part(self.valid_loader, self.model, self.run_dir, config.save_val, self.epoch)

        self.primary_metric = primary_metric
        self.scheduler.step(primary_metric)

        if primary_metric > self.best_metric:
            logger.success("Best model renewed")
            self.best_metric = primary_metric
            save_checkpoint(self.model, self.run_dir) # Saves as checkpoint_best.ckpt
            return True
        return False

    @error_wrap
    def _test(self, tag: str) -> None:
        """Test"""
        logger.info(f"Test with {tag} model from epoch {self.epoch}")

        if tag == "best":
            checkpoint_path = self.run_dir / "checkpoints" / "checkpoint_best.ckpt" # Correct path
            if not checkpoint_path.exists():
                logger.warning(f"best checkpoint not found in {checkpoint_path}. Skipping test.")
                return
            
            checkpoint = torch.load(checkpoint_path)
            state_dict = checkpoint.get('model_state_dict')
            if state_dict:
                model_to_load = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
                model_to_load.load_state_dict(state_dict)
            else:
                logger.error("Could not find a valid state_dict in the checkpoint.")

        test_part(self.test_loader, self.model, self.run_dir, True, self.epoch, test_mode=True)


def main() -> None:
    """execution entry point"""
    parse_args_for_train_script()
    trainer = Trainer()
    trainer()


if __name__ == "__main__":
    main()
