# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import inspect
import logging
import os
import json

import hydra
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger, CSVLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.strategies.ddp import DDPStrategy
from omegaconf import DictConfig, OmegaConf
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from solo.args.linear import parse_cfg
from solo.data.classification_dataloader import build_grid_dataset, prepare_data
from solo.methods.base import BaseMethod
from solo.methods.linear import LinearModel
from solo.utils.auto_resumer import AutoResumer
from solo.utils.checkpointer import Checkpointer
from solo.utils.misc import make_contiguous
from lightning.pytorch.callbacks import ModelCheckpoint
from solo.utils.custom_callbacks import EpochTensorBoardLogger, SimplifiedProgressBar

try:
    from solo.data.dali_dataloader import ClassificationDALIDataModule
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True

@torch.no_grad()
def extract_features(model, dataloader, desc="Extracting Features", debug=False): # <-- Added debug param
    """Runs data through the backbone once and caches the features in memory."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.backbone.to(device)
    model.backbone.eval()
    
    all_features = []
    all_targets = []
    
    for i, (X, y) in enumerate(dataloader):
        if debug and i >= 2:  # <-- Added early exit for debugging
            break

        X = X.to(device)
        
        # Apply channel last if configured
        if not getattr(model, 'no_channel_last', False):
            if X.ndim == 5:
                X = X.to(memory_format=torch.channels_last_3d)
            elif X.ndim == 4:
                X = X.to(memory_format=torch.channels_last)
                
        # 1. Use Autocast to simulate '16-mixed' precision on the GPU
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            feats = model.backbone(X)
            
        # 2. Move to CPU and immediately cast down to FP16 to save system RAM
        feats = feats.cpu().to(torch.float16)
        
        all_features.append(feats)
        all_targets.append(y)
        
    model.backbone.cpu() # Return backbone to CPU for Trainer
    
    all_features = torch.cat(all_features, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    dataset = TensorDataset(all_features, all_targets)
    
    is_train = isinstance(dataloader.sampler, torch.utils.data.RandomSampler)
    return DataLoader(
        dataset,
        batch_size=dataloader.batch_size,
        shuffle=is_train,
        num_workers=0, 
        pin_memory=True
    )

@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    # hydra doesn't allow us to add new keys for "safety"
    # set_struct(..., False) disables this behavior and allows us to add more parameters
    # without making the user specify every single thing about the model
    OmegaConf.set_struct(cfg, False)
    cfg = parse_cfg(cfg)
    if cfg.data.get("task", "classification") == "classification" and cfg.data.get("dataset_kwargs", {}).get("target") == "synthesizability":
        cfg.data.num_classes = 1
    print(cfg)

    if "_target_" in cfg.backbone:
        backbone = hydra.utils.instantiate(cfg.backbone)
    else:
        # Fallback to the old dictionary lookup method
        backbone_model = BaseMethod._BACKBONES[cfg.backbone.name]
        backbone = backbone_model(method=cfg.pretrain_method, **cfg.backbone.kwargs)

    backbone_name = cfg.backbone.get("name", "") or cfg.backbone.get("_target_", "")
    
    if hasattr(backbone, "fc"):
        backbone.fc = nn.Identity()
    if hasattr(backbone, "heads"):
        backbone.heads.head = nn.Identity()

    ckpt_path = cfg.pretrained_feature_extractor
    
    # Only load if a path is actually provided
    if ckpt_path is not None:
        ckpt_str = str(ckpt_path).lower()
        if ckpt_str == "imagenet":
            logging.info("Using ImageNet pre-trained weights. Bypassing state_dict loading.")
        elif ckpt_str.endswith("offline_features.pt") or ckpt_str.endswith("offline_features_full.pt"):
            logging.info("Bypassing backbone weight loading because an offline cache is provided.")
        else:
            assert ckpt_path.endswith(".ckpt") or ckpt_path.endswith(".pth") or ckpt_path.endswith(".pt")

            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
            new_state = {}
            for k, v in state.items():
                if "encoder" in k:
                    k = k.replace("encoder", "backbone")
                    logging.warning("You are using an older checkpoint...")
                if "backbone." in k:
                    k = k.replace("backbone.", "")
                new_state[k] = v
            backbone.load_state_dict(new_state, strict=False)
            logging.info(f"Loaded {ckpt_path}")
    else:
        logging.info("No checkpoint provided. Initializing backbone with random weights.")

    # check if mixup or cutmix is enabled
    mixup_func = None
    mixup_active = cfg.mixup > 0 or cfg.cutmix > 0
    if mixup_active:
        logging.info("Mixup activated")
        mixup_func = Mixup(
            mixup_alpha=cfg.mixup,
            cutmix_alpha=cfg.cutmix,
            cutmix_minmax=None,
            prob=1.0,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=cfg.label_smoothing,
            num_classes=cfg.data.num_classes,
        )
        # smoothing is handled with mixup label transform
        loss_func = SoftTargetCrossEntropy()
    elif cfg.label_smoothing > 0:
        loss_func = LabelSmoothingCrossEntropy(smoothing=cfg.label_smoothing)
    else:
        if cfg.data.get("task", "classification") == "regression":
            loss_func = torch.nn.MSELoss()
        else:
            if cfg.data.num_classes == 1:
                # Insert your calculated ratio here
                weight = torch.tensor([1]) 
                loss_func = nn.BCEWithLogitsLoss(pos_weight=weight)
            else:
                loss_func = torch.nn.CrossEntropyLoss()
    model = LinearModel(backbone, loss_func=loss_func, mixup_func=mixup_func, cfg=cfg)
    make_contiguous(model)
    # can provide up to ~20% speed up
    if not cfg.performance.disable_channel_last:
        if any(isinstance(m, nn.Conv3d) for m in model.modules()):
            model = model.to(memory_format=torch.channels_last_3d)
        else:
            model = model.to(memory_format=torch.channels_last)

    if cfg.data.format == "dali":
        val_data_format = "image_folder"
    else:
        val_data_format = cfg.data.format
    
    if cfg.get("cache_features", False):
        logging.info("Caching features: Overriding train_transform with val_transform to ensure deterministic representations.")
        cfg.train_transform = cfg.val_transform

    offline_cache_path = None
    if cfg.get("cache_features", False) and cfg.pretrained_feature_extractor is not None:
        job_dir = os.path.dirname(cfg.pretrained_feature_extractor)
        possible_cache = os.path.join(job_dir, "offline_features_full.pt")
        if os.path.exists(possible_cache):
            offline_cache_path = possible_cache
        else:
            possible_cache = os.path.join(job_dir, "offline_features.pt")
            if os.path.exists(possible_cache):
                offline_cache_path = possible_cache
    # Check if we are using handcrafted features
    use_handcrafted = cfg.data.get("dataset_kwargs", {}).get("use_handcrafted", False)
    should_bypass_disk = (offline_cache_path is not None) or use_handcrafted

    # Cached-feature loading needs dataset metadata but not the source grids.
    if should_bypass_disk:
        logging.info("Bypassing os.path.exists to speed up dataset initialization (Cache or Handcrafted enabled).")
        original_exists = os.path.exists
        def fast_exists(path):
            if isinstance(path, str) and path.endswith('.h5py'):
                return True
            return original_exists(path)
        os.path.exists = fast_exists

    train_loader, val_loader = prepare_data(
        cfg.data.dataset,
        train_data_path=cfg.data.train_path,
        val_data_path=cfg.data.val_path,
        data_format=val_data_format,
        batch_size=cfg.optimizer.batch_size,
        num_workers=cfg.data.num_workers,
        auto_augment=cfg.auto_augment,
        linear_probe_subset_size=cfg.data.get('linear_probe_subset_size', None),
        cfg_train_transform=cfg.get("train_transform", None),
        cfg_val_transform=cfg.get("val_transform", None),
        dataset_kwargs=cfg.data.get("dataset_kwargs", None),
    )

    # Restore the original os.path.exists function immediately after
    if should_bypass_disk:
        os.path.exists = original_exists
    
    if cfg.data.get("task", "classification") == "classification" and cfg.data.get("dataset_kwargs", {}).get("target") == "synthesizability":
        logging.info("Calculating dynamic pos_weight from the training subset...")
        
        ds = train_loader.dataset
        # Safely handle the Subset wrapper natively used in classification_dataloader.py
        if isinstance(ds, torch.utils.data.Subset):
            base_ds = ds.dataset
            indices = ds.indices
        else:
            base_ds = ds
            indices = range(len(ds))

        pos_count = 0
        neg_count = 0

        for orig_idx in indices:
            # Grid datasets expose their target field through target_key.
            target_key = getattr(base_ds, "target_key", "kH_C18")
            label = base_ds.files[orig_idx].get(target_key, float('nan'))
            if label == 1.0:
                pos_count += 1
            elif label == 0.0:
                neg_count += 1

        if pos_count > 0:
            dynamic_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32)
            # Re-initialize the loss function on the model with the calculated weight
            model.loss_func = nn.BCEWithLogitsLoss(pos_weight=dynamic_weight)
            logging.info(f"Dynamically set pos_weight to {dynamic_weight.item():.4f} (Pos: {pos_count}, Neg: {neg_count})")
        else:
            logging.warning("No positive samples found in the training subset! Pos_weight cannot be calculated.")

    def get_offline_cached_dataloader(dataloader, cache_dict):
        from torch.utils.data import TensorDataset, DataLoader
        ds = dataloader.dataset
        all_feats, all_targets = [], []

        # Safely handle the Subset wrapper natively used in classification_dataloader.py
        if isinstance(ds, torch.utils.data.Subset):
            base_ds = ds.dataset
            indices = ds.indices
        else:
            base_ds = ds
            indices = range(len(ds))

        for orig_idx in indices:
            sample_id_key = getattr(base_ds, "sample_id_key", "zeolite")
            target_key = getattr(base_ds, "target_key", "kH_C18")
            record = base_ds.files[orig_idx]
            name = record[sample_id_key]
            
            # A missing cache key means there is no precomputed representation.
            if name not in cache_dict:
                continue
            
            # Grid datasets expose their target field through target_key.
            label = record.get(target_key, float('nan'))
            all_feats.append(cache_dict[name].squeeze())
            if cfg.data.get("task", "classification") == "classification":
                all_targets.append(torch.tensor(int(label), dtype=torch.long))
            else:
                all_targets.append(torch.tensor(label, dtype=torch.float32))

        all_feats = torch.stack(all_feats, dim=0)
        all_targets = torch.stack(all_targets, dim=0)
        
        tensor_ds = TensorDataset(all_feats, all_targets)
        is_train = isinstance(dataloader.sampler, torch.utils.data.RandomSampler)
        
        return DataLoader(
            tensor_ds, 
            batch_size=dataloader.batch_size, 
            shuffle=is_train, 
            num_workers=0, 
            pin_memory=True
        )

    if cfg.get("cache_features", False) and cfg.data.format != "dali":
        logging.info("Starting feature caching phase...")

        if offline_cache_path is not None:
            logging.info(f"Loading offline cached features from {offline_cache_path}")
            cache_dict = torch.load(offline_cache_path, map_location="cpu", weights_only=False)

            sample_feat = next(iter(cache_dict.values())).squeeze()
            actual_dim = sample_feat.shape[-1]
            
            if model.classifier.in_features != actual_dim:
                logging.info(f"Adjusting classifier dimension from {model.classifier.in_features} to {actual_dim} to match cached features.")
                if model.task == "regression":
                    model.classifier = nn.Linear(actual_dim, 1)
                else:
                    model.classifier = nn.Linear(actual_dim, cfg.data.num_classes)
            
            train_loader = get_offline_cached_dataloader(train_loader, cache_dict)
            val_loader = get_offline_cached_dataloader(val_loader, cache_dict)
            
            # Save the cache reference to the model so the Test evaluation phase can also use it
            model.offline_cache_dict = cache_dict 
            model.finetune = False
        else:
            train_loader = extract_features(model, train_loader, desc="Caching Train")
            val_loader = extract_features(model, val_loader, desc="Caching Val")
            model.finetune = False

    if cfg.data.format == "dali":
        assert (
            _dali_avaliable
        ), "Dali is not currently avaiable, please install it first with pip3 install .[dali]."

        assert not cfg.auto_augment, "Auto augmentation is not supported with Dali."

        dali_datamodule = ClassificationDALIDataModule(
            dataset=cfg.data.dataset,
            train_data_path=cfg.data.train_path,
            val_data_path=cfg.data.val_path,
            num_workers=cfg.data.num_workers,
            batch_size=cfg.optimizer.batch_size,
            data_fraction=cfg.data.fraction,
            dali_device=cfg.dali.device,
        )

        # use normal torchvision dataloader for validation to save memory
        dali_datamodule.val_dataloader = lambda: val_loader

    current_run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # 1.7 will deprecate resume_from_checkpoint, but for the moment
    # the argument is the same, but we need to pass it as ckpt_path to trainer.fit
    ckpt_path, wandb_run_id = None, None
    if cfg.auto_resume.enabled and cfg.resume_from_checkpoint is None:
        auto_resumer = AutoResumer(
            checkpoint_dir=current_run_dir,
            max_hours=cfg.auto_resume.max_hours,
        )
        resume_from_checkpoint, wandb_run_id = auto_resumer.find_checkpoint(cfg)
        if resume_from_checkpoint is not None:
            print(
                "Resuming from previous checkpoint that matches specifications:",
                f"'{resume_from_checkpoint}'",
            )
            ckpt_path = resume_from_checkpoint
    elif cfg.resume_from_checkpoint is not None:
        ckpt_path = cfg.resume_from_checkpoint
        del cfg.resume_from_checkpoint

    if ckpt_path is not None:
        # Resume writing to the previous run's exact directory
        output_dir = os.path.dirname(ckpt_path)
    else:
        # Fresh run: use Hydra's newly generated directory
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    callbacks = []
    callbacks.append(SimplifiedProgressBar())
    callbacks.append(EpochTensorBoardLogger())

    if cfg.checkpoint.enabled:
        # Use Lightning's built-in ModelCheckpoint
        ckpt = ModelCheckpoint(
            dirpath=output_dir,
            monitor=cfg.checkpoint.monitor,
            mode=cfg.checkpoint.mode,
            save_top_k=cfg.checkpoint.save_top_k,
            save_last=cfg.checkpoint.save_last,
            # Saves a checkpoint every N epochs
            every_n_epochs=cfg.checkpoint.frequency,
        )
        callbacks.append(ckpt)

    loggers = []

    # wandb logging
    wandb_logger = None
    if cfg.wandb.enabled:
        wandb_logger = WandbLogger(
            name=cfg.name,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            offline=cfg.wandb.offline,
            resume="allow" if wandb_run_id else None,
            id=wandb_run_id,
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))
    else:
        # Fallback to TensorBoard
        print("Tensorboard Logging")
        wandb_logger = TensorBoardLogger(
            save_dir=output_dir,
            name="tensorboard_logs",
            version=wandb_run_id
        )
        if wandb_run_id is None:
            wandb_run_id = wandb_logger.version

    # lr logging
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    loggers.append(wandb_logger)
    loggers.append(CSVLogger(save_dir=output_dir, name="csv_logs"))

    trainer_kwargs = OmegaConf.to_container(cfg)
    # we only want to pass in valid Trainer args, the rest may be user specific
    valid_kwargs = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {name: trainer_kwargs[name] for name in valid_kwargs if name in trainer_kwargs}
    trainer_kwargs.update(
        {
            "logger": loggers,
            "callbacks": callbacks,
            "enable_checkpointing": cfg.checkpoint.enabled,
            "enable_progress_bar": False,
            "strategy": DDPStrategy(find_unused_parameters=False)
            if cfg.strategy == "ddp"
            else cfg.strategy,
            "profiler": "simple",
        }
    )
    is_debug = cfg.get("debug", False)
    if is_debug:
        logging.info("🐛 DEBUG MODE ENABLED: Limiting pipeline to 2 batches (fast_dev_run).")
        trainer_kwargs["fast_dev_run"] = 2
    cfg.wandb_run_id = wandb_run_id
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=4)
        
    trainer = Trainer(**trainer_kwargs)

    if cfg.data.format == "dali":
        trainer.fit(model, ckpt_path=ckpt_path, datamodule=dali_datamodule)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    if cfg.data.dataset in ("zeolites", "mofs"):
        # Create Test Dataset
        T_test = hydra.utils.instantiate(cfg.val_transform) 
        
        # Check if we should bypass disk for the test set
        use_handcrafted = cfg.data.get("dataset_kwargs", {}).get("use_handcrafted", False)
        should_bypass_disk_test = hasattr(model, "offline_cache_dict") or use_handcrafted

        if should_bypass_disk_test:
            original_exists = os.path.exists
            def fast_exists(path):
                if isinstance(path, str) and path.endswith('.h5py'):
                    return True
                return original_exists(path)
            os.path.exists = fast_exists

        test_dataset = build_grid_dataset(
            dataset=cfg.data.dataset,
            root_dir=cfg.data.val_path,
            is_train=False,
            is_val=False,
            transform=T_test,
            dataset_kwargs=cfg.data.dataset_kwargs,
        )
        
        # Restore immediately after
        if should_bypass_disk_test:
            os.path.exists = original_exists
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=cfg.optimizer.batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            drop_last=False
        )

        if cfg.get("cache_features", False):
            model.caching_features = True
            
            # Check if offline cache was previously loaded into memory
            if hasattr(model, "offline_cache_dict"):
                print("Applying offline cached features to Test Set...")
                test_loader = get_offline_cached_dataloader(test_loader, model.offline_cache_dict)
            else:
                test_loader = extract_features(model, test_loader, desc="Caching Test", debug=is_debug)
        
        print(f"Running evaluation on Test Set: {len(test_dataset)} samples")
        trainer.test(model, dataloaders=test_loader, ckpt_path="best")


if __name__ == "__main__":
    main()
