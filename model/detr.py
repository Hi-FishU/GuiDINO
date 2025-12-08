import torch
from torch import nn

from model.backbone import DINOv3BackboneWrapper
from model.positionembedding import PositionEmbeddingSine
from model.transformer import Transformer


class DETRHead(nn.Module):
    def __init__(
        self,
        backbone: DINOv3BackboneWrapper,
        num_classes: int,
        hidden_dim: int = 256,
        nheads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_queries: int = 100,
        aux_loss: bool = True
    ):
        super().__init__()
        self.backbone = backbone

        with torch.no_grad():
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.input_proj = nn.Conv2d(
            backbone.embed_dim, hidden_dim, kernel_size=1)
        self.position_embedding = PositionEmbeddingSine(
            hidden_dim // 2, normalize=True)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu"
        )

        # Prediction heads
        self.class_embed = nn.Linear(
            hidden_dim, num_classes + 1)  # +1 for "no-object"
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.aux_loss = aux_loss

    def forward(self, samples: torch.Tensor, sizes=None):
        """
        samples: (B, 3, H, W) preprocessed images for DINOv3
        sizes:   optional list of (orig_h, orig_w) for post-processing
        Returns:
          outputs: dict with 'pred_logits' (B, Q, K+1) and 'pred_boxes' (B, Q, 4 in [0,1], cxcywh)
                   and optionally 'aux_outputs' for intermediate decoder layers (if aux_loss=True).
        """
        features, mask = self.backbone(
            samples, sizes)  # (B, Cb, H', W'), (B, H', W')
        pos = self.position_embedding((features, mask))  # (B, C, H', W')
        src = self.input_proj(features)                  # (B, C, H', W')

        hs = self.transformer(src, mask,
                              self.query_embed.weight, pos)[0]
        # B, C, H, W = src.shape
        # src_flat = src.flatten(2).permute(2, 0, 1)  # (HW, B, C)
        # pos_flat = pos.flatten(2).permute(2, 0, 1)  # (HW, B, C)
        # mask_flat = mask.flatten(1)                 # (B, HW)

        # # Prepare target (decoder) sequence: queries (Q, B, C)
        # query_embed = self.query_embed.weight.unsqueeze(
        #     1).repeat(1, B, 1)  # (Q, B, C)
        # tgt = torch.zeros_like(query_embed)  # (Q, B, C)

        # memory = self.transformer.encoder(
        #     src_flat + pos_flat, src_key_padding_mask=mask_flat)
        # hs = self.transformer.decoder(
        #     tgt + query_embed, memory + pos_flat, memory_key_padding_mask=mask_flat)
        # # hs: (num_decoder_layers, Q, B, C)
        # hs = hs.transpose(1, 2)  # (num_decoder_layers, B, Q, C)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(
            hs).sigmoid()  # normalized [0,1], cxcywh

        out = {
            "pred_logits": outputs_class[-1],     # (B, Q, K+1)
            "pred_boxes": outputs_coord[-1],      # (B, Q, 4)
        }
        if self.aux_loss:
            out["aux_outputs"] = [
                {"pred_logits": a, "pred_boxes": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers - 1):
            in_d = input_dim if i == 0 else hidden_dim
            layers += [nn.Linear(in_d, hidden_dim), nn.ReLU(inplace=True)]
        layers += [nn.Linear(hidden_dim, output_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
