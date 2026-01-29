"""Training script for PlaylistLSTM model."""

import time
from pathlib import Path
from typing import Optional, Set, Dict, Any

import torch
import torch.nn as nn
from tqdm import tqdm

from .build_vocab import Vocabulary, load_splits
from .config import Config, get_device
from .datasets import create_sequence_dataloader
from .models import PlaylistLSTM, Item2Vec
from .train_item2vec import load_item2vec


def train_lstm(
    config: Config,
    vocab: Optional[Vocabulary] = None,
    train_pids: Optional[Set[int]] = None,
    val_pids: Optional[Set[int]] = None,
    item2vec: Optional[Item2Vec] = None,
) -> PlaylistLSTM:
    """
    Train PlaylistLSTM model for sequence prediction.

    Args:
        config: Configuration object.
        vocab: Optional pre-loaded vocabulary.
        train_pids: Optional pre-loaded training playlist IDs.
        val_pids: Optional pre-loaded validation playlist IDs.
        item2vec: Optional Item2Vec model for embedding initialization.

    Returns:
        Trained PlaylistLSTM model.
    """
    device = get_device()
    print(f"Using device: {device}")

    # Load vocabulary if not provided
    if vocab is None:
        print(f"Loading vocabulary from {config.paths.vocab_path}")
        vocab = Vocabulary.load(config.paths.vocab_path)
    print(f"Vocabulary size: {vocab.size}")

    # Load splits if not provided
    if train_pids is None or val_pids is None:
        print("Loading training splits...")
        train_pids_loaded, val_pids_loaded, _ = load_splits(
            config.paths.train_pids_path,
            config.paths.val_pids_path,
            config.paths.test_pids_path,
        )
        train_pids = train_pids or train_pids_loaded
        val_pids = val_pids or val_pids_loaded
    print(f"Training on {len(train_pids)} playlists, validating on {len(val_pids)}")

    # Load Item2Vec if requested
    if config.lstm.init_from_item2vec and item2vec is None:
        if config.paths.item2vec_path.exists():
            print(f"Loading Item2Vec from {config.paths.item2vec_path}")
            item2vec = load_item2vec(config.paths.item2vec_path, device)
        else:
            print("Item2Vec model not found, skipping embedding initialization")

    # Create model
    model = PlaylistLSTM(
        vocab_size=vocab.size,
        embedding_dim=config.item2vec.embedding_dim,
        hidden_dim=config.lstm.hidden_dim,
        num_layers=config.lstm.num_layers,
        dropout=config.lstm.dropout,
        padding_idx=vocab.pad_idx,
        tie_weights=config.lstm.tie_weights,
    )

    # Initialize from Item2Vec
    if item2vec is not None:
        model.init_from_item2vec(item2vec)

    model = model.to(device)

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader = create_sequence_dataloader(
        mpd_dir=config.paths.mpd_dir,
        vocab=vocab,
        playlist_pids=train_pids,
        batch_size=config.lstm.batch_size,
        num_workers=config.lstm.num_workers,
        shuffle=True,
        max_seq_length=config.lstm.max_seq_length,
        augment=True,
    )

    val_loader = create_sequence_dataloader(
        mpd_dir=config.paths.mpd_dir,
        vocab=vocab,
        playlist_pids=val_pids,
        batch_size=config.lstm.batch_size,
        num_workers=config.lstm.num_workers,
        shuffle=False,
        max_seq_length=config.lstm.max_seq_length,
        augment=False,
    )

    # Loss function (ignore padding)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lstm.learning_rate,
        weight_decay=config.lstm.weight_decay,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        verbose=True,
    )

    # Create output directory
    config.paths.lstm_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.lstm.epochs):
        print(f"\nEpoch {epoch + 1}/{config.lstm.epochs}")

        # Training
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            gradient_clip=config.lstm.gradient_clip,
        )
        print(f"Train Loss: {train_loss:.4f}, Perplexity: {torch.exp(torch.tensor(train_loss)):.2f}")

        # Validation
        val_loss = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )
        print(f"Val Loss: {val_loss:.4f}, Perplexity: {torch.exp(torch.tensor(val_loss)):.2f}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save checkpoint
        checkpoint_path = config.paths.lstm_path.parent / f"lstm_epoch{epoch + 1}.pt"
        save_checkpoint(model, optimizer, epoch, val_loss, config, checkpoint_path)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            save_checkpoint(model, optimizer, epoch, val_loss, config, config.paths.lstm_path)
            print(f"Saved best model to {config.paths.lstm_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{config.lstm.patience}")

            if patience_counter >= config.lstm.patience:
                print("Early stopping triggered!")
                break

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Load best model
    model = load_lstm(config.paths.lstm_path, device)
    return model


def train_epoch(
    model: PlaylistLSTM,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float = 5.0,
) -> float:
    """
    Train for one epoch.

    Returns:
        Average training loss.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for inputs, targets, lengths in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        lengths = lengths.to(device)

        # Forward pass
        logits, _ = model(inputs, lengths)

        # Compute loss
        # Reshape for CrossEntropyLoss: (batch * seq_len, vocab_size)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if num_batches % 50 == 0:
            pbar.set_postfix({"loss": f"{total_loss / num_batches:.4f}"})

    return total_loss / max(num_batches, 1)


def validate(
    model: PlaylistLSTM,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Validate model.

    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets, lengths in tqdm(dataloader, desc="Validating"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)

            logits, _ = model(inputs, lengths)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def save_checkpoint(
    model: PlaylistLSTM,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    config: Config,
    path: Path,
) -> None:
    """Save model checkpoint."""
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": {
            "vocab_size": model.vocab_size,
            "embedding_dim": model.embedding_dim,
            "hidden_dim": model.hidden_dim,
            "num_layers": model.num_layers,
            "tie_weights": model.tie_weights,
        },
    }, path)


def load_lstm(path: Path | str, device: Optional[torch.device] = None) -> PlaylistLSTM:
    """
    Load a trained PlaylistLSTM model.

    Args:
        path: Path to model checkpoint.
        device: Device to load model to.

    Returns:
        Loaded PlaylistLSTM model.
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(path, map_location=device)
    model_config = checkpoint["config"]

    model = PlaylistLSTM(
        vocab_size=model_config["vocab_size"],
        embedding_dim=model_config["embedding_dim"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        tie_weights=model_config.get("tie_weights", True),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from loss."""
    return torch.exp(torch.tensor(loss)).item()
