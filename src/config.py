"""Centralized configuration for Spotify Playlist Recommender."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class PathConfig:
    """Paths for data and models."""
    mpd_dir: Path = Path("data/mpd")
    vocab_path: Path = Path("artifacts/vocab.json")
    item2vec_path: Path = Path("artifacts/item2vec.pt")
    lstm_path: Path = Path("artifacts/lstm.pt")
    faiss_index_path: Path = Path("artifacts/faiss.index")
    log_dir: Path = Path("logs")
    train_pids_path: Path = Path("artifacts/train_pids.json")
    val_pids_path: Path = Path("artifacts/val_pids.json")
    test_pids_path: Path = Path("artifacts/test_pids.json")

    def __post_init__(self):
        self.mpd_dir = Path(self.mpd_dir)
        self.vocab_path = Path(self.vocab_path)
        self.item2vec_path = Path(self.item2vec_path)
        self.lstm_path = Path(self.lstm_path)
        self.faiss_index_path = Path(self.faiss_index_path)
        self.log_dir = Path(self.log_dir)
        self.train_pids_path = Path(self.train_pids_path)
        self.val_pids_path = Path(self.val_pids_path)
        self.test_pids_path = Path(self.test_pids_path)

    def ensure_dirs(self):
        """Create necessary directories."""
        self.vocab_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class VocabConfig:
    """Vocabulary building configuration."""
    min_track_freq: int = 5
    min_playlist_length: int = 5
    max_vocab_size: int = 500_000


@dataclass
class Item2VecConfig:
    """Item2Vec model configuration."""
    embedding_dim: int = 128
    window_size: int = 5
    num_negatives: int = 5
    learning_rate: float = 0.025
    min_learning_rate: float = 0.0001
    epochs: int = 5
    batch_size: int = 4096
    subsample_threshold: float = 1e-4
    num_workers: int = 4


@dataclass
class LSTMConfig:
    """PlaylistLSTM model configuration."""
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    epochs: int = 20
    batch_size: int = 64
    max_seq_length: int = 100
    gradient_clip: float = 5.0
    patience: int = 5
    num_workers: int = 4
    tie_weights: bool = True
    init_from_item2vec: bool = True


@dataclass
class InferenceConfig:
    """Inference configuration."""
    num_candidates: int = 500
    num_recommendations: int = 100
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    diversity_penalty: float = 0.0
    nprobe: int = 10  # FAISS IVF search parameter


@dataclass
class Config:
    """Master configuration combining all configs."""
    paths: PathConfig = field(default_factory=PathConfig)
    vocab: VocabConfig = field(default_factory=VocabConfig)
    item2vec: Item2VecConfig = field(default_factory=Item2VecConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(
            paths=PathConfig(**data.get("paths", {})),
            vocab=VocabConfig(**data.get("vocab", {})),
            item2vec=Item2VecConfig(**data.get("item2vec", {})),
            lstm=LSTMConfig(**data.get("lstm", {})),
            inference=InferenceConfig(**data.get("inference", {})),
            seed=data.get("seed", 42),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        from dataclasses import asdict

        data = {
            "paths": {k: str(v) for k, v in asdict(self.paths).items()},
            "vocab": asdict(self.vocab),
            "item2vec": asdict(self.item2vec),
            "lstm": asdict(self.lstm),
            "inference": asdict(self.inference),
            "seed": self.seed,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


def get_device():
    """Get the best available device (MPS > CUDA > CPU)."""
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
