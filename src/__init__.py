"""Spotify Playlist Recommender - Two-stage recommendation system using Item2Vec and LSTM."""

from .config import Config, PathConfig, VocabConfig, Item2VecConfig, LSTMConfig, InferenceConfig, get_device
from .build_vocab import Vocabulary, build_vocabulary, split_playlists
from .utils_mpd import Track, Playlist, iter_playlists, iter_mpd_slices
from .models import Item2Vec, PlaylistLSTM
from .train_item2vec import train_item2vec, load_item2vec
from .train_seq_model import train_lstm, load_lstm
from .infer_candidates import CandidateRetriever, LSTMRanker, two_stage_recommend, compute_metrics
from .generate_playlist import generate_playlist, generate_with_two_stage
from .datasets import Item2VecDataset, PlaylistSequenceDataset

__version__ = "0.1.0"

__all__ = [
    # Config
    "Config",
    "PathConfig",
    "VocabConfig",
    "Item2VecConfig",
    "LSTMConfig",
    "InferenceConfig",
    "get_device",
    # Data structures
    "Track",
    "Playlist",
    "Vocabulary",
    # Data loading
    "iter_playlists",
    "iter_mpd_slices",
    "build_vocabulary",
    "split_playlists",
    # Datasets
    "Item2VecDataset",
    "PlaylistSequenceDataset",
    # Models
    "Item2Vec",
    "PlaylistLSTM",
    # Training
    "train_item2vec",
    "load_item2vec",
    "train_lstm",
    "load_lstm",
    # Inference
    "CandidateRetriever",
    "LSTMRanker",
    "two_stage_recommend",
    "compute_metrics",
    # Generation
    "generate_playlist",
    "generate_with_two_stage",
]
