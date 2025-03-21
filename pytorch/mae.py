import torch
from einops.layers.torch import Rearrange
import torch.nn as nn
from vit import PatchEmbedding, Block



class MaskedAutoEncoder(nn.Module):
    def __init__(self, emb_size=1024, decoder_emb_size=512, patch_size=16, num_head=16, 
                 encoder_num_layers=24, decoder_num_layers=8, in_channels=3, img_size=224):
        super().__init__()
        # Pass the in_channels parameter to PatchEmbedding
        self.patch_embed = PatchEmbedding(emb_size=emb_size, in_channels=in_channels, patch_size=patch_size, img_size=img_size)
        self.decoder_embed = nn.Linear(emb_size, decoder_emb_size)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size)**2 + 1, decoder_emb_size), requires_grad=False)
        self.decoder_pred = nn.Linear(decoder_emb_size, patch_size**2 * in_channels, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_emb_size))
        self.encoder_transformer = nn.Sequential(*[Block(emb_size, num_head) for _ in range(encoder_num_layers)])
        self.decoder_transformer = nn.Sequential(*[Block(decoder_emb_size, num_head) for _ in range(decoder_num_layers)])
        # IMPORTANT: Update the project module to use in_channels (instead of 3)
        self.project = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=patch_size**2 * in_channels, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def random_masking(self, x, mask_ratio):
        """
        x: (B, T, C)
        Random masking to create randomly shuffled unmasked patches.
        """
        B, T, D = x.shape  
        len_keep = int(T * (1 - mask_ratio))
        # Create noise of shape (B, T)
        noise = torch.rand(B, T, device=x.device)
        # Sort noise and get shuffle indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # Keep the first len_keep indices
        ids_keep = ids_shuffle[:, :len_keep]
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # Create binary mask: 0 for keep, 1 for remove
        mask = torch.ones([B, T], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle the mask to original order
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x, mask, ids_restore

    def encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        cls_token = x[:, :1, :]
        x = x[:, 1:, :]
        x, mask, restore_id = self.random_masking(x, mask_ratio)
        x = torch.cat((cls_token, x), dim=1)
        x = self.encoder_transformer(x)
        return x, mask, restore_id
        
    def decoder(self, x, restore_id):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], restore_id.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=restore_id.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        # Add positional embedding
        x = x + self.decoder_pos_embed
        x = self.decoder_transformer(x)
        # Predictor projection
        x = self.decoder_pred(x)
        # Remove class token
        x = x[:, 1:, :]
        return x

    def loss(self, imgs, pred, mask):
        """
        imgs: [N, C, H, W]
        pred: [N, L, patch*patch*C]
        mask: [N, L] where 0 is keep, 1 is remove
        """
        target = self.project(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, img):
        mask_ratio = 0.75
        x, mask, restore_ids = self.encoder(img, mask_ratio)
        pred = self.decoder(x, restore_ids)
        loss = self.loss(img, pred, mask)
        
        return loss, pred, mask