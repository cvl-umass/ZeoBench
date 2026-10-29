import torch
import torch.nn as nn
import torchvision.models as models

class ViT3DSequenceWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        
    def forward(self, x):
        """
        Expects x of shape: (B, 1, D, H, W)
        D must be divisible by 3.
        """
        if x.dim() == 4:
            x = x.unsqueeze(1) # Ensure 5D tensor: (B, 1, D, H, W)
            
        B, C, D, H, W = x.shape
        assert C == 1, "Expected 1-channel volume"
        assert D % 3 == 0, f"Depth D ({D}) must be divisible by 3"
        
        # 1. Chunk the depth dimension into groups of 3
        # Reshape to (B, D/3, 3, H, W)
        chunks = D // 3
        x = x.view(B, chunks, 3, H, W)
        
        # 2. Flatten Batch and Chunks to pass through 2D ViT patch embedder
        # Shape becomes (B * chunks, 3, H, W)
        x = x.reshape(B * chunks, 3, H, W)
        
        # 3. Extract patches (generates 196 patches per 3-slice chunk for 224x224)
        x = self.vit._process_input(x) # Shape: (B * chunks, N_patches, hidden_dim)
        N_patches = x.shape[1]
        hidden_dim = x.shape[2]
        
        # 4. Reshape back to separate Chunks and Patches
        x = x.view(B, chunks, N_patches, hidden_dim)
        
        # 5. Extract and apply 2D Positional Embeddings
        pos_embed = self.vit.encoder.pos_embedding
        cls_pos_embed = pos_embed[:, :1, :]
        patch_pos_embed = pos_embed[:, 1:, :]
        
        # Broadcasting handles the chunk dimension seamlessly
        x = x + patch_pos_embed.unsqueeze(1) 
        
        # 6. Flatten chunks into the final sequence
        x = x.reshape(B, chunks * N_patches, hidden_dim)
        
        # 7. Prepare and append the CLS token
        cls_token = self.vit.class_token.expand(B, -1, -1)
        cls_token = cls_token + cls_pos_embed
        x = torch.cat([cls_token, x], dim=1)
        
        # 8. Pass through the Transformer
        x = self.vit.encoder.dropout(x)
        x = self.vit.encoder.layers(x)
        x = self.vit.encoder.ln(x)
        
        return x[:, 0] # Return CLS representation