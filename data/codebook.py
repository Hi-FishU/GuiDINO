from dataclasses import dataclass

@dataclass
class CodebookConfig:
    k: int = 512
    pca_dim: int = 256
    use_pca: bool = True
    token_sample_per_image: int = 128   # random tokens per image to avoid bias
    max_tokens_for_fit: int = 800_000   # cap for PCA/KMeans fit
    batch_size_for_ipca: int = 8192
    kmeans_batch_size: int = 8192
    random_seed: int = 0
