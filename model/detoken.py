import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from torch import nn

from data.codebook import CodebookConfig
from model.backbone import DINOv3BackboneWrapper
from model.positionembedding import PositionEmbeddingSine
from utils.codebook_tool import (iter_images_from_loader, l2_normalize,
                                 top_p_mean)


class Token_Processor(nn.Module):
    def __init__(self,
                 backbone: DINOv3BackboneWrapper,
                 hidden_dim: int = 256):
        super().__init__()

        self.backbone = backbone
        self.input_proj = nn.Conv2d(
            backbone.embed_dim, hidden_dim, kernel_size=1)
        self.position_embedding = PositionEmbeddingSine(
            hidden_dim // 2, normalize=True)

    @torch.no_grad()
    def token_forward(self, samples: torch.Tensor):
        """
        samples: (B, 3, H, W) preprocessed images for DINOv3
        Returns:
          features: (B, hidden_dim, H_feat, W_feat) feature maps
          pos: (B, hidden_dim, H_feat, W_feat) position embeddings
        """
        features = self.backbone(samples)
        src = self.input_proj(features)
        pos = self.position_embedding(src)

        return src, pos

    def forward(self, samples: torch.Tensor, sizes=None):
        src, pos = self.token_forward(samples)

        return src + pos

class KMeansCodebook(nn.Module):
    def __init__(self, cfg: CodebookConfig):
        super().__init__()
        self.cfg = cfg
        self.ipca: Optional[IncrementalPCA] = None
        self.kmeans: Optional[MiniBatchKMeans] = None
        # L2-normalized centroids in PCA space (or raw space)

        self.centroids_: Optional[np.ndarray] = None

        np.random.seed(cfg.random_seed)

    def _sample_tokens(self, tokens: torch.Tensor) -> np.ndarray:
        """
        tokens: torch.Tensor [B, Np, D] on GPU
        Returns sampled tokens as np.ndarray [Ns, D] on CPU
        """
        B, Np, D = tokens.shape
        t = tokens.reshape(B * Np, D)
        # Randomly sample a fixed count per image to reduce spatial/patient bias.
        # Alternative: stratify or uniform spatial sampling; start simple.
        per_img = self.cfg.token_sample_per_image
        idx_list = []
        for b in range(B):
            start = b * Np
            end = (b + 1) * Np
            n = min(per_img, Np)
            idx = np.random.choice(np.arange(start, end),
                                   size=n, replace=False)
            idx_list.append(idx)
        idx = np.concatenate(idx_list, axis=0)
        sampled = t[idx].detach().cpu().float().numpy()
        return sampled

    def fit(self, extractor: Token_Processor, normal_loader, device: str = "cuda") -> None:
        cfg = self.cfg

        # 1) Collect sampled tokens (streaming) up to max_tokens_for_fit
        chunks: List[np.ndarray] = []
        total = 0
        extractor.to(device)

        for x in iter_images_from_loader(normal_loader):
            x = x.to(device, non_blocking=True)
            toks = extractor(x)  # [B, Np, D]
            samp = self._sample_tokens(toks)  # [Ns, D]
            chunks.append(samp)
            total += samp.shape[0]
            if total >= cfg.max_tokens_for_fit:
                break

        X = np.concatenate(chunks, axis=0)
        X = X[:cfg.max_tokens_for_fit]
        # L2 normalize in original space before PCA
        X = l2_normalize(X)

        # 2) Fit IncrementalPCA (optional)
        if cfg.use_pca:
            self.ipca = IncrementalPCA(
                n_components=cfg.pca_dim, batch_size=cfg.batch_size_for_ipca)
            # partial_fit in batches
            for i in range(0, X.shape[0], cfg.batch_size_for_ipca):
                self.ipca.partial_fit(X[i:i + cfg.batch_size_for_ipca])
            Xp = self.ipca.transform(X).astype(np.float32)
            Xp = l2_normalize(Xp)
        else:
            Xp = X.astype(np.float32)

        # 3) Fit MiniBatchKMeans
        self.kmeans = MiniBatchKMeans(
            n_clusters=cfg.k,
            batch_size=cfg.kmeans_batch_size,
            init="k-means++",
            n_init="auto",
            reassignment_ratio=0.01,
            random_state=cfg.random_seed,
            verbose=0,
        )

        # Fit in a streaming-like way
        for i in range(0, Xp.shape[0], cfg.kmeans_batch_size):
            self.kmeans.partial_fit(Xp[i:i + cfg.kmeans_batch_size])

        C = self.kmeans.cluster_centers_.astype(np.float32)
        C = l2_normalize(C)
        self.centroids_ = C

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        cfg_path = os.path.join(path, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(self.cfg.__dict__, f, indent=2)

        np.save(os.path.join(path, "centroids.npy"), self.centroids_)

        if self.ipca is not None:
            # save PCA components for reproducibility
            np.save(os.path.join(path, "ipca_components.npy"),
                    self.ipca.components_.astype(np.float32))
            np.save(os.path.join(path, "ipca_mean.npy"),
                    self.ipca.mean_.astype(np.float32))
            np.save(os.path.join(path, "ipca_var.npy"),
                    self.ipca.var_.astype(np.float32))
            with open(os.path.join(path, "ipca_meta.json"), "w") as f:
                json.dump(
                    {"n_components": self.ipca.n_components_}, f, indent=2)

    @staticmethod
    def load(path: str) -> "KMeansCodebook":
        with open(os.path.join(path, "config.json"), "r") as f:
            cfg = CodebookConfig(**json.load(f))

        obj = KMeansCodebook(cfg)
        obj.centroids_ = np.load(os.path.join(path, "centroids.npy"))

        # Restore IncrementalPCA (enough fields for transform)
        comp_path = os.path.join(path, "ipca_components.npy")
        if os.path.exists(comp_path):
            comps = np.load(comp_path)
            mean = np.load(os.path.join(path, "ipca_mean.npy"))
            var = np.load(os.path.join(path, "ipca_var.npy"))
            with open(os.path.join(path, "ipca_meta.json"), "r") as f:
                meta = json.load(f)

            ipca = IncrementalPCA(n_components=int(meta["n_components"]))
            # Hack: set learned attributes needed for transform
            ipca.components_ = comps
            ipca.mean_ = mean
            ipca.var_ = var
            ipca.n_features_in_ = comps.shape[1]
            ipca.n_components_ = comps.shape[0]
            ipca.explained_variance_ = var[:ipca.n_components_] if var.ndim == 1 else np.diag(var)[
                :ipca.n_components_]
            obj.ipca = ipca

        return obj

    @torch.no_grad()
    def score_image_tokens(
        self,
        extractor: Token_Processor,
        x: torch.Tensor,
        device: str = "cuda",
        top_p: float = 0.02,
    ) -> Dict[str, np.ndarray]:
        """
        Returns:
          token_anomaly: [Np] float32, higher = more abnormal
          token_code: [Np] int32 assigned centroid index
          image_score: scalar (top-p mean)
        """
        assert self.centroids_ is not None, "Codebook not fitted/loaded."
        extractor.to(device)
        x = x.to(device, non_blocking=True)
        toks = extractor(x)  # [B, Np, D]
        if toks.shape[0] != 1:
            raise ValueError(
                "score_image_tokens expects batch size = 1 for simplicity.")

        z = toks[0].detach().cpu().float().numpy()  # [Np, D]
        z = l2_normalize(z)

        if self.ipca is not None:
            z = self.ipca.transform(z).astype(np.float32)
            z = l2_normalize(z)

        # cosine sim with centroids (since all are L2-normalized)
        sim = z @ self.centroids_.T  # [Np, K]
        code = sim.argmax(axis=1).astype(np.int32)
        best = sim.max(axis=1).astype(np.float32)
        anomaly = (1.0 - best).astype(np.float32)

        return {
            "token_anomaly": anomaly,
            "token_code": code,
            "image_score": np.array([top_p_mean(anomaly, p=top_p)], dtype=np.float32),
        }
