# In solo/methods/mae_monai_3d.py

from solo.methods.base import BaseMethod
from solo.losses.mae import mae_loss_func
import torch

from solo.backbones.custom.mae_monai import MaskedAutoEncoderViT as MonaiMaskedAutoEncoderViT
from solo.utils.metrics import weighted_mean

# --- ADD THIS NEW PATCHED CLASS ---
class PatchedMaskedAutoEncoderViT(MonaiMaskedAutoEncoderViT):
    def __init__(self, **kwargs):
        # The __init__ is identical to the parent, so we just call it
        super().__init__(**kwargs)

    def forward(self, x, masking_ratio: float | None = None):
        x = self.patch_embedding(x)
        x, selected_indices, mask = self._masking(x, masking_ratio=masking_ratio)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.blocks(x)

        # decoder
        x = self.decoder_embed(x)

        mask_tokens = self.mask_tokens.to(dtype=x.dtype)

        x_ = mask_tokens.repeat(x.shape[0], mask.shape[1], 1)
        x_[torch.arange(x.shape[0]).unsqueeze(-1), selected_indices] = x[:, 1:, :]  # no cls token
        x_ = x_ + self.decoder_pos_embedding
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = self.decoder_blocks(x)
        x = self.decoder_pred(x)

        x = x[:, 1:, :]
        return x, mask

class MAEMonai3D(BaseMethod):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = self.backbone
        self.mask_ratio = cfg.method_kwargs.mask_ratio

    @property
    def learnable_params(self) -> list:
        return [
            {"name": "model", "params": self.model.parameters()},
            {
                "name": "classifier",
                "params": self.classifier.parameters(),
                "lr": self.classifier_lr,
                "weight_decay": 0,
            },
        ]

    def forward(self, X: torch.Tensor) -> dict:
        # Used for validation/linear probe
        feats = self.forward_features(X)
        logits = self.classifier(feats.detach())
        return {"logits": logits, "feats": feats}

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Helper to extract features for the linear probe
        x = self.model.patch_embedding(x)
        cls_tokens = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.model.blocks(x)
        return x[:, 0]

    def training_step(self, batch, batch_idx):
        # 1. Run BaseMethod training step 
        # This calculates the online classification/regression loss on the features
        out = super().training_step(batch, batch_idx)
        
        # 'out["loss"]' here is the Online Regression Loss (if task="regression")
        online_reg_loss = out["loss"]

        _, X, _ = batch
        images = X[0]  # Only one view needed for MAE

        # 2. MAE Forward Pass
        predictions, mask = self.model(images, masking_ratio=self.mask_ratio)

        # 3. Reconstruction Loss
        recon_loss = mae_loss_func(
            imgs=images,
            pred=predictions,
            mask=mask,
            patch_size=self.model.patch_size[0],
            use_patchify_3d=True,
        )

        # 4. Logging
        # on_step=True: Logs per-step values (good for debugging spikes)
        # on_epoch=True: Logs the average at the end of the epoch (required for your Custom Callback)
        self.log_dict(
            {
                "train_recon_loss": recon_loss, 
                "train_online_loss": online_reg_loss 
            },
            on_epoch=True, 
            on_step=True, 
            sync_dist=True
        )

        # 5. Total Loss
        # We optimize for reconstruction, but the online head optimizes for regression simultaneously
        return recon_loss + online_reg_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        # 1. Calculate online regression metrics (Standard BaseMethod behavior)
        metrics = super().validation_step(batch, batch_idx, dataloader_idx)
        
        # 2. Calculate Validation Reconstruction Loss
        X, targets = batch
        predictions, mask = self.model(X, masking_ratio=self.mask_ratio)
        
        val_recon_loss = mae_loss_func(
            imgs=X,
            pred=predictions,
            mask=mask,
            patch_size=self.model.patch_size[0],
            use_patchify_3d=True
        )
        
        # Add to the metrics list so we can average it later
        metrics["val_recon_loss"] = val_recon_loss
        
        # IMPORTANT: BaseMethod automatically appends 'metrics' to self.validation_step_outputs.
        # However, super() created a new dict. We need to ensure our new key is stored.
        # Since super() returns the dict that WAS appended, modifying 'metrics' here 
        # might not update the reference inside 'validation_step_outputs' if it was a shallow copy?
        # Safe fix: Update the last entry in the list explicitly.
        if self.validation_step_outputs:
            self.validation_step_outputs[-1]["val_recon_loss"] = val_recon_loss

        return metrics

    def on_validation_epoch_end(self):
        # 1. BaseMethod averages val_loss, val_acc1, etc.
        super().on_validation_epoch_end()
        
        # 2. We average our custom Validation Reconstruction Loss
        if not self.validation_step_outputs:
            return

        # Calculate standard online evaluation metrics
        val_loss = weighted_mean(self.validation_step_outputs, "val_loss", "batch_size")
        log = {"val_loss": val_loss} # This is the Online Head Loss (Regression MSE)

        # Calculate reconstruction metrics
        if "val_recon_loss" in self.validation_step_outputs[0]:
            val_recon_loss = weighted_mean(self.validation_step_outputs, "val_recon_loss", "batch_size")
            log["val_recon_loss"] = val_recon_loss

        # Log everything
        self.log_dict(log, sync_dist=True)
        
        # Clean up
        self.validation_step_outputs.clear()
