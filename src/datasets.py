"""PyTorch datasets for training Item2Vec and PlaylistLSTM models."""

import random
from pathlib import Path
from typing import Iterator, List, Tuple, Set, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset
from torch.nn.utils.rnn import pad_sequence

from .build_vocab import Vocabulary, PAD_IDX, BOS_IDX, EOS_IDX
from .utils_mpd import iter_playlists, get_track_uris_from_playlist


class Item2VecDataset(IterableDataset):
    """
    Streaming dataset for Item2Vec skip-gram training.

    Generates (center, context, negatives) tuples from playlist co-occurrences.
    Uses subsampling of frequent tracks and dynamic window sizes.
    """

    def __init__(
        self,
        mpd_dir: Path | str,
        vocab: Vocabulary,
        train_pids: Set[int],
        window_size: int = 5,
        num_negatives: int = 5,
        subsample_threshold: float = 1e-4,
    ):
        """
        Args:
            mpd_dir: Path to MPD directory.
            vocab: Vocabulary object.
            train_pids: Set of playlist IDs to use for training.
            window_size: Maximum context window size on each side.
            num_negatives: Number of negative samples per positive pair.
            subsample_threshold: Threshold for subsampling frequent tracks.
        """
        self.mpd_dir = Path(mpd_dir)
        self.vocab = vocab
        self.train_pids = train_pids
        self.window_size = window_size
        self.num_negatives = num_negatives

        # Precompute sampling distributions
        self.subsample_probs = vocab.get_subsampling_probs(subsample_threshold)
        self.negative_weights = vocab.get_sampling_weights(power=0.75)

        # Create sampling table for efficient negative sampling
        self._build_sampling_table()

    def _build_sampling_table(self, table_size: int = 100_000_000):
        """Build unigram table for efficient negative sampling."""
        # Scale weights and create cumulative distribution
        vocab_size = self.vocab.size
        counts = np.zeros(vocab_size)
        for uri, idx in self.vocab.track2idx.items():
            if idx >= 4:  # Skip special tokens
                counts[idx] = self.vocab.track_counts.get(uri, 0) ** 0.75

        # Normalize
        total = counts.sum()
        if total > 0:
            probs = counts / total
        else:
            probs = np.ones(vocab_size) / vocab_size

        self.neg_probs = probs

    def _sample_negatives(self, exclude: Set[int]) -> List[int]:
        """Sample negative indices excluding the given set."""
        negatives = []
        while len(negatives) < self.num_negatives:
            neg = np.random.choice(self.vocab.size, p=self.neg_probs)
            if neg not in exclude and neg >= 4:  # Skip special tokens
                negatives.append(neg)
        return negatives

    def _should_keep(self, idx: int) -> bool:
        """Determine if token should be kept based on subsampling."""
        return random.random() < self.subsample_probs[idx]

    def __iter__(self) -> Iterator[Tuple[int, int, List[int]]]:
        """
        Iterate through skip-gram pairs with negative samples.

        Yields:
            Tuple of (center_idx, context_idx, negative_indices).
        """
        worker_info = torch.utils.data.get_worker_info()

        for playlist in iter_playlists(self.mpd_dir):
            # Filter to training playlists
            if playlist.pid not in self.train_pids:
                continue

            # Shard across workers
            if worker_info is not None:
                if playlist.pid % worker_info.num_workers != worker_info.id:
                    continue

            # Encode playlist tracks
            track_uris = get_track_uris_from_playlist(playlist)
            indices = self.vocab.encode(track_uris, add_special=False)

            # Filter out UNK tokens and apply subsampling
            filtered_indices = [
                idx for idx in indices
                if idx != self.vocab.unk_idx and self._should_keep(idx)
            ]

            if len(filtered_indices) < 2:
                continue

            # Generate skip-gram pairs
            for i, center_idx in enumerate(filtered_indices):
                # Dynamic window size (uniform sample from [1, window_size])
                actual_window = random.randint(1, self.window_size)

                # Get context window
                start = max(0, i - actual_window)
                end = min(len(filtered_indices), i + actual_window + 1)

                context_indices = set(filtered_indices[start:i] + filtered_indices[i + 1:end])

                for context_idx in context_indices:
                    # Sample negatives
                    exclude = {center_idx, context_idx} | context_indices
                    negatives = self._sample_negatives(exclude)

                    yield center_idx, context_idx, negatives


class PlaylistSequenceDataset(Dataset):
    """
    Random-access dataset for PlaylistLSTM training.

    Stores encoded sequences for efficient shuffling.
    Returns (input_seq, target_seq) pairs for next-token prediction.
    """

    def __init__(
        self,
        mpd_dir: Path | str,
        vocab: Vocabulary,
        playlist_pids: Set[int],
        max_seq_length: int = 100,
        augment: bool = True,
        min_seq_length: int = 5,
    ):
        """
        Args:
            mpd_dir: Path to MPD directory.
            vocab: Vocabulary object.
            playlist_pids: Set of playlist IDs to include.
            max_seq_length: Maximum sequence length.
            augment: Whether to use random subsequences for augmentation.
            min_seq_length: Minimum sequence length to include.
        """
        self.vocab = vocab
        self.max_seq_length = max_seq_length
        self.augment = augment
        self.min_seq_length = min_seq_length

        # Load and encode all sequences
        self.sequences: List[List[int]] = []
        self._load_sequences(mpd_dir, playlist_pids)

    def _load_sequences(self, mpd_dir: Path | str, playlist_pids: Set[int]) -> None:
        """Load and encode sequences from MPD."""
        from tqdm import tqdm

        print("Loading playlist sequences...")
        for playlist in tqdm(iter_playlists(mpd_dir), desc="Loading sequences"):
            if playlist.pid not in playlist_pids:
                continue

            track_uris = get_track_uris_from_playlist(playlist)
            indices = self.vocab.encode(track_uris, add_special=False)

            # Filter out UNK tokens
            indices = [idx for idx in indices if idx != self.vocab.unk_idx]

            if len(indices) >= self.min_seq_length:
                self.sequences.append(indices)

        print(f"Loaded {len(self.sequences)} sequences")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence pair for training.

        Returns:
            Tuple of (input_sequence, target_sequence) tensors.
            Input has BOS prepended, target is the original sequence.
        """
        sequence = self.sequences[idx]

        # Data augmentation: random subsequence
        if self.augment and len(sequence) > self.max_seq_length:
            start = random.randint(0, len(sequence) - self.max_seq_length)
            sequence = sequence[start:start + self.max_seq_length]
        elif len(sequence) > self.max_seq_length:
            sequence = sequence[:self.max_seq_length]

        # Input: BOS + sequence[:-1], Target: sequence
        input_seq = [BOS_IDX] + sequence[:-1]
        target_seq = sequence

        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long),
        )


def collate_item2vec(
    batch: List[Tuple[int, int, List[int]]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for Item2Vec batches.

    Args:
        batch: List of (center, context, negatives) tuples.

    Returns:
        Tuple of (centers, contexts, negatives) tensors.
    """
    centers = torch.tensor([b[0] for b in batch], dtype=torch.long)
    contexts = torch.tensor([b[1] for b in batch], dtype=torch.long)
    negatives = torch.tensor([b[2] for b in batch], dtype=torch.long)

    return centers, contexts, negatives


def collate_sequences(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    pad_idx: int = PAD_IDX,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for sequence batches with padding.

    Args:
        batch: List of (input_seq, target_seq) tuples.
        pad_idx: Padding token index.

    Returns:
        Tuple of (padded_inputs, padded_targets, lengths) tensors.
    """
    inputs, targets = zip(*batch)

    # Get lengths before padding
    lengths = torch.tensor([len(seq) for seq in inputs], dtype=torch.long)

    # Pad sequences
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_idx)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=pad_idx)

    return padded_inputs, padded_targets, lengths


def create_item2vec_dataloader(
    mpd_dir: Path | str,
    vocab: Vocabulary,
    train_pids: Set[int],
    batch_size: int = 4096,
    num_workers: int = 4,
    **dataset_kwargs,
) -> torch.utils.data.DataLoader:
    """Create DataLoader for Item2Vec training."""
    dataset = Item2VecDataset(
        mpd_dir=mpd_dir,
        vocab=vocab,
        train_pids=train_pids,
        **dataset_kwargs,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_item2vec,
        num_workers=num_workers,
        pin_memory=True,
    )


def create_sequence_dataloader(
    mpd_dir: Path | str,
    vocab: Vocabulary,
    playlist_pids: Set[int],
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: bool = True,
    **dataset_kwargs,
) -> torch.utils.data.DataLoader:
    """Create DataLoader for LSTM training."""
    dataset = PlaylistSequenceDataset(
        mpd_dir=mpd_dir,
        vocab=vocab,
        playlist_pids=playlist_pids,
        **dataset_kwargs,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_sequences(b, vocab.pad_idx),
        num_workers=num_workers,
        pin_memory=True,
    )
