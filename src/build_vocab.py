"""Vocabulary construction for the Spotify Playlist Recommender."""

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
from tqdm import tqdm

from .config import VocabConfig
from .utils_mpd import iter_playlists, get_track_uris_from_playlist


# Special token indices
PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]


@dataclass
class Vocabulary:
    """Track vocabulary with special tokens and frequency information."""
    track2idx: Dict[str, int]
    idx2track: Dict[int, str]
    track_counts: Dict[str, int]
    total_count: int = 0

    @property
    def size(self) -> int:
        """Return vocabulary size including special tokens."""
        return len(self.track2idx)

    @property
    def pad_idx(self) -> int:
        return PAD_IDX

    @property
    def unk_idx(self) -> int:
        return UNK_IDX

    @property
    def bos_idx(self) -> int:
        return BOS_IDX

    @property
    def eos_idx(self) -> int:
        return EOS_IDX

    def encode(self, track_uris: List[str], add_special: bool = False) -> List[int]:
        """
        Encode track URIs to indices.

        Args:
            track_uris: List of track URIs.
            add_special: If True, prepend BOS and append EOS.

        Returns:
            List of token indices.
        """
        indices = [self.track2idx.get(uri, UNK_IDX) for uri in track_uris]
        if add_special:
            indices = [BOS_IDX] + indices + [EOS_IDX]
        return indices

    def decode(self, indices: List[int], skip_special: bool = True) -> List[str]:
        """
        Decode indices to track URIs.

        Args:
            indices: List of token indices.
            skip_special: If True, skip special tokens in output.

        Returns:
            List of track URIs.
        """
        special_indices = {PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX} if skip_special else set()
        return [
            self.idx2track.get(idx, "<UNK>")
            for idx in indices
            if idx not in special_indices
        ]

    def get_sampling_weights(self, power: float = 0.75) -> np.ndarray:
        """
        Get negative sampling weights (frequency^power distribution).

        Args:
            power: Exponent for frequency weighting (0.75 is Word2Vec default).

        Returns:
            Normalized sampling weights for each index.
        """
        weights = np.zeros(self.size, dtype=np.float32)
        for uri, idx in self.track2idx.items():
            if idx >= len(SPECIAL_TOKENS):  # Skip special tokens
                weights[idx] = self.track_counts.get(uri, 0) ** power
        weights /= weights.sum()
        return weights

    def get_subsampling_probs(self, threshold: float = 1e-4) -> np.ndarray:
        """
        Get subsampling probabilities for frequent tracks (Word2Vec style).

        Args:
            threshold: Subsampling threshold.

        Returns:
            Array of keep probabilities for each index.
        """
        probs = np.ones(self.size, dtype=np.float32)
        if self.total_count == 0:
            return probs

        for uri, idx in self.track2idx.items():
            if idx >= len(SPECIAL_TOKENS):
                freq = self.track_counts.get(uri, 0) / self.total_count
                if freq > 0:
                    # Word2Vec subsampling formula
                    probs[idx] = min(1.0, (np.sqrt(freq / threshold) + 1) * (threshold / freq))
        return probs

    def save(self, path: Path | str) -> None:
        """Save vocabulary to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "track2idx": self.track2idx,
            "track_counts": self.track_counts,
            "total_count": self.total_count,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: Path | str) -> "Vocabulary":
        """Load vocabulary from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        track2idx = data["track2idx"]
        idx2track = {int(idx): uri for uri, idx in track2idx.items()}
        # Ensure special tokens are in idx2track
        for i, token in enumerate(SPECIAL_TOKENS):
            idx2track[i] = token

        return cls(
            track2idx=track2idx,
            idx2track=idx2track,
            track_counts=data["track_counts"],
            total_count=data.get("total_count", sum(data["track_counts"].values())),
        )

    def __contains__(self, track_uri: str) -> bool:
        return track_uri in self.track2idx


def build_vocabulary(mpd_dir: Path | str, config: VocabConfig) -> Vocabulary:
    """
    Build vocabulary from MPD data using two-pass approach.

    Pass 1: Count all track occurrences
    Pass 2: Filter by frequency and build vocabulary

    Args:
        mpd_dir: Path to MPD directory.
        config: Vocabulary configuration.

    Returns:
        Constructed Vocabulary object.
    """
    mpd_dir = Path(mpd_dir)

    # Pass 1: Count track frequencies
    print("Pass 1: Counting track frequencies...")
    track_counts: Counter = Counter()

    for playlist in tqdm(iter_playlists(mpd_dir), desc="Counting tracks"):
        if len(playlist.tracks) >= config.min_playlist_length:
            for track in playlist.tracks:
                track_counts[track.track_uri] += 1

    print(f"Found {len(track_counts)} unique tracks")

    # Pass 2: Filter and build vocabulary
    print(f"Pass 2: Filtering tracks with min_freq={config.min_track_freq}...")

    # Filter by minimum frequency
    filtered_tracks = [
        (uri, count) for uri, count in track_counts.items()
        if count >= config.min_track_freq
    ]

    # Sort by frequency (descending) and limit to max_vocab_size
    filtered_tracks.sort(key=lambda x: x[1], reverse=True)
    if len(filtered_tracks) > config.max_vocab_size - len(SPECIAL_TOKENS):
        filtered_tracks = filtered_tracks[:config.max_vocab_size - len(SPECIAL_TOKENS)]

    print(f"Vocabulary size after filtering: {len(filtered_tracks)}")

    # Build mappings
    track2idx: Dict[str, int] = {}
    idx2track: Dict[int, str] = {}

    # Add special tokens first
    for i, token in enumerate(SPECIAL_TOKENS):
        track2idx[token] = i
        idx2track[i] = token

    # Add filtered tracks
    for uri, _ in filtered_tracks:
        idx = len(track2idx)
        track2idx[uri] = idx
        idx2track[idx] = uri

    # Calculate total count for subsampling
    total_count = sum(count for _, count in filtered_tracks)

    return Vocabulary(
        track2idx=track2idx,
        idx2track=idx2track,
        track_counts=dict(filtered_tracks),
        total_count=total_count,
    )


def split_playlists(
    mpd_dir: Path | str,
    vocab: Vocabulary,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    min_known_tracks: int = 5,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split playlists into train/val/test sets.

    Only includes playlists that have enough known tracks in vocabulary.

    Args:
        mpd_dir: Path to MPD directory.
        vocab: Vocabulary object.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        seed: Random seed for reproducibility.
        min_known_tracks: Minimum tracks that must be in vocabulary.

    Returns:
        Tuple of (train_pids, val_pids, test_pids).
    """
    mpd_dir = Path(mpd_dir)
    random.seed(seed)

    # Collect eligible playlist IDs
    eligible_pids: List[int] = []

    print("Collecting eligible playlists...")
    for playlist in tqdm(iter_playlists(mpd_dir), desc="Filtering playlists"):
        track_uris = get_track_uris_from_playlist(playlist)
        known_tracks = sum(1 for uri in track_uris if uri in vocab)
        if known_tracks >= min_known_tracks:
            eligible_pids.append(playlist.pid)

    print(f"Found {len(eligible_pids)} eligible playlists")

    # Shuffle and split
    random.shuffle(eligible_pids)

    n_total = len(eligible_pids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_pids = eligible_pids[:n_train]
    val_pids = eligible_pids[n_train:n_train + n_val]
    test_pids = eligible_pids[n_train + n_val:]

    print(f"Split: train={len(train_pids)}, val={len(val_pids)}, test={len(test_pids)}")

    return train_pids, val_pids, test_pids


def save_splits(
    train_pids: List[int],
    val_pids: List[int],
    test_pids: List[int],
    train_path: Path | str,
    val_path: Path | str,
    test_path: Path | str,
) -> None:
    """Save playlist ID splits to JSON files."""
    for pids, path in [
        (train_pids, train_path),
        (val_pids, val_path),
        (test_pids, test_path),
    ]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(pids, f)


def load_splits(
    train_path: Path | str,
    val_path: Path | str,
    test_path: Path | str,
) -> Tuple[Set[int], Set[int], Set[int]]:
    """Load playlist ID splits from JSON files."""
    splits = []
    for path in [train_path, val_path, test_path]:
        with open(path, "r") as f:
            splits.append(set(json.load(f)))
    return tuple(splits)
