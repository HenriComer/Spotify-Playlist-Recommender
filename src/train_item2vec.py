"""Training script for Item2Vec model."""

import time
from pathlib import Path
from typing import Optional, Set

import torch
from tqdm import tqdm

from .build_vocab import Vocabulary, load_splits
from .config import Config, get_device
from .datasets import create_item2vec_dataloader
from .models import Item2Vec


def train_item2vec(
    config: Config,
    vocab: Optional[Vocabulary] = None,
    train_pids: Optional[Set[int]] = None,
) -> Item2Vec:
    """
    Train Item2Vec model with skip-gram negative sampling.

    Args:
        config: Configuration object.
        vocab: Optional pre-loaded vocabulary.
        train_pids: Optional pre-loaded training playlist IDs.

    Returns:
        Trained Item2Vec model.
    """
    device = get_device()
    print(f"Using device: {device}")

    # Load vocabulary if not provided
    if vocab is None:
        print(f"Loading vocabulary from {config.paths.vocab_path}")
        vocab = Vocabulary.load(config.paths.vocab_path)
    print(f"Vocabulary size: {vocab.size}")

    # Load training playlist IDs if not provided
    if train_pids is None:
        print(f"Loading training splits...")
        train_pids, _, _ = load_splits(
            config.paths.train_pids_path,
            config.paths.val_pids_path,
            config.paths.test_pids_path,
        )
    print(f"Training on {len(train_pids)} playlists")

    # Create model
    model = Item2Vec(
        vocab_size=vocab.size,
        embedding_dim=config.item2vec.embedding_dim,
        padding_idx=vocab.pad_idx,
    )
    model = model.to(device)

    # Use SparseAdam for sparse embeddings
    optimizer = torch.optim.SparseAdam(
        model.parameters(),
        lr=config.item2vec.learning_rate,
    )

    # Learning rate schedule (linear decay)
    total_steps_estimate = len(train_pids) * 50 * config.item2vec.epochs  # rough estimate
    lr_start = config.item2vec.learning_rate
    lr_end = config.item2vec.min_learning_rate

    # Create output directory
    config.paths.item2vec_path.parent.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_loss = float("inf")

    for epoch in range(config.item2vec.epochs):
        print(f"\nEpoch {epoch + 1}/{config.item2vec.epochs}")

        # Create dataloader for this epoch (streaming)
        dataloader = create_item2vec_dataloader(
            mpd_dir=config.paths.mpd_dir,
            vocab=vocab,
            train_pids=train_pids,
            batch_size=config.item2vec.batch_size,
            num_workers=config.item2vec.num_workers,
            window_size=config.item2vec.window_size,
            num_negatives=config.item2vec.num_negatives,
            subsample_threshold=config.item2vec.subsample_threshold,
        )

        model.train()
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        for batch_idx, (centers, contexts, negatives) in enumerate(pbar):
            # Move to device
            centers = centers.to(device)
            contexts = contexts.to(device)
            negatives = negatives.to(device)

            # Linear learning rate decay
            progress = global_step / total_steps_estimate
            lr = lr_start - (lr_start - lr_end) * min(progress, 1.0)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Forward pass
            loss = model(centers, contexts, negatives)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Update progress bar
            if batch_idx % 100 == 0:
                avg_loss = epoch_loss / num_batches
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{lr:.6f}",
                })

        # Epoch statistics
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch + 1} - Loss: {avg_epoch_loss:.4f}, Time: {elapsed:.1f}s")

        # Save checkpoint
        checkpoint_path = config.paths.item2vec_path.parent / f"item2vec_epoch{epoch + 1}.pt"
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_epoch_loss,
            "config": {
                "vocab_size": vocab.size,
                "embedding_dim": config.item2vec.embedding_dim,
            },
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "loss": avg_epoch_loss,
                "config": {
                    "vocab_size": vocab.size,
                    "embedding_dim": config.item2vec.embedding_dim,
                },
            }, config.paths.item2vec_path)
            print(f"Saved best model to {config.paths.item2vec_path}")

    print("\nTraining complete!")
    return model


def load_item2vec(path: Path | str, device: Optional[torch.device] = None) -> Item2Vec:
    """
    Load a trained Item2Vec model.

    Args:
        path: Path to model checkpoint.
        device: Device to load model to.

    Returns:
        Loaded Item2Vec model.
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(path, map_location=device)
    model_config = checkpoint["config"]

    model = Item2Vec(
        vocab_size=model_config["vocab_size"],
        embedding_dim=model_config["embedding_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def evaluate_item2vec(
    model: Item2Vec,
    vocab: Vocabulary,
    sample_tracks: int = 5,
    k: int = 10,
) -> None:
    """
    Evaluate Item2Vec by showing nearest neighbors for sample tracks.

    Args:
        model: Trained Item2Vec model.
        vocab: Vocabulary object.
        sample_tracks: Number of tracks to sample.
        k: Number of neighbors to show.
    """
    import random

    # Sample some track indices (excluding special tokens)
    track_indices = list(range(4, min(vocab.size, 1000)))
    sample_indices = random.sample(track_indices, min(sample_tracks, len(track_indices)))

    print("\nNearest neighbors for sample tracks:")
    print("=" * 60)

    for idx in sample_indices:
        query_uri = vocab.idx2track.get(idx, "<UNK>")
        print(f"\nQuery: {query_uri}")

        neighbors = model.get_similar(idx, k=k)
        for neighbor_idx, score in neighbors:
            neighbor_uri = vocab.idx2track.get(neighbor_idx, "<UNK>")
            print(f"  {score:.4f}: {neighbor_uri}")
