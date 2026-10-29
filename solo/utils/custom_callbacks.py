# solo/utils/custom_callbacks.py
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger

class SimplifiedProgressBar(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        
        # Pull the classification loss
        class_loss = metrics.get("train_class_loss", 0.0)
        log_str = f"Epoch {epoch}: Class Loss: {class_loss:.4f}"

        # Known pretraining metric keys across the methods
        pretrain_keys = ["dino_loss", "train_nce_loss", "train_neg_cos_sim"]
        
        # Dynamically find and append pretraining losses
        for key, value in metrics.items():
            if (key.endswith("_loss") or key in pretrain_keys) and key not in ["train_class_loss", "val_loss", "momentum_val_loss"]:
                log_str += f", {key}: {value:.4f}"

        if "train_acc1" in metrics:
            log_str += f", Train Acc1: {metrics['train_acc1']:.2f}"
            
        print(log_str, flush=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        
        loss = metrics.get("val_loss", 0.0)
        log_str = f"Epoch {epoch}: Val Loss: {loss:.4f}"
        
        if "val_acc1" in metrics:
            log_str += f", Val Acc1: {metrics['val_acc1']:.2f}"
            
        print(log_str, flush=True)

class EpochTensorBoardLogger(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        self._log_epoch_metrics(trainer, "train")

    def on_validation_epoch_end(self, trainer, pl_module):
        self._log_epoch_metrics(trainer, "val")
    
    def on_test_epoch_end(self, trainer, pl_module):
        self._log_epoch_metrics(trainer, "test")
    
    def _log_epoch_metrics(self, trainer, stage):
        if isinstance(trainer.logger, TensorBoardLogger):
            metrics = trainer.callback_metrics
            epoch = trainer.current_epoch
            for k, v in metrics.items():
                if stage in k and isinstance(v, (int, float, torch.Tensor)):
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    trainer.logger.experiment.add_scalar(
                        f"{k}_per_epoch", v, global_step=epoch
                    )
