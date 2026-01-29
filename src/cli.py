"""Command-line interface for Spotify Playlist Recommender."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def build_vocab_command(args):
    """Build vocabulary from MPD data."""
    from .build_vocab import build_vocabulary, split_playlists, save_splits
    from .config import Config, VocabConfig

    print(f"Building vocabulary from {args.mpd_dir}")

    # Load or create config
    if args.config:
        config = Config.from_yaml(args.config)
        config.paths.mpd_dir = Path(args.mpd_dir)
    else:
        config = Config()
        config.paths.mpd_dir = Path(args.mpd_dir)

    if args.output:
        config.paths.vocab_path = Path(args.output)

    if args.min_freq:
        config.vocab.min_track_freq = args.min_freq

    # Build vocabulary
    vocab = build_vocabulary(config.paths.mpd_dir, config.vocab)
    vocab.save(config.paths.vocab_path)
    print(f"Saved vocabulary to {config.paths.vocab_path}")
    print(f"Vocabulary size: {vocab.size}")

    # Create train/val/test splits
    if not args.skip_splits:
        print("\nCreating train/val/test splits...")
        train_pids, val_pids, test_pids = split_playlists(
            config.paths.mpd_dir,
            vocab,
            seed=config.seed,
        )
        save_splits(
            train_pids, val_pids, test_pids,
            config.paths.train_pids_path,
            config.paths.val_pids_path,
            config.paths.test_pids_path,
        )
        print(f"Saved splits to {config.paths.train_pids_path.parent}")


def train_item2vec_command(args):
    """Train Item2Vec model."""
    from .config import Config
    from .train_item2vec import train_item2vec, evaluate_item2vec
    from .build_vocab import Vocabulary

    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    # Override config with command line args
    if args.epochs:
        config.item2vec.epochs = args.epochs
    if args.batch_size:
        config.item2vec.batch_size = args.batch_size
    if args.embedding_dim:
        config.item2vec.embedding_dim = args.embedding_dim

    # Train model
    model = train_item2vec(config)

    # Evaluate
    if not args.skip_eval:
        vocab = Vocabulary.load(config.paths.vocab_path)
        evaluate_item2vec(model, vocab)


def train_lstm_command(args):
    """Train PlaylistLSTM model."""
    from .config import Config
    from .train_seq_model import train_lstm

    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    # Override config with command line args
    if args.epochs:
        config.lstm.epochs = args.epochs
    if args.batch_size:
        config.lstm.batch_size = args.batch_size
    if args.hidden_dim:
        config.lstm.hidden_dim = args.hidden_dim
    if args.no_pretrain:
        config.lstm.init_from_item2vec = False

    # Train model
    train_lstm(config)


def recommend_command(args):
    """Generate playlist recommendations."""
    from .config import Config, get_device
    from .build_vocab import Vocabulary
    from .train_item2vec import load_item2vec
    from .train_seq_model import load_lstm
    from .infer_candidates import (
        CandidateRetriever, LSTMRanker, two_stage_recommend,
        build_retriever_from_item2vec,
    )
    from .generate_playlist import generate_playlist

    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    device = get_device()
    print(f"Using device: {device}")

    # Load vocab
    print("Loading vocabulary...")
    vocab = Vocabulary.load(config.paths.vocab_path)

    # Parse seed tracks
    seed_uris = [s.strip() for s in args.seeds.split(",")]
    print(f"Seed tracks: {len(seed_uris)}")

    # Load models based on mode
    if args.mode == "lstm":
        # Pure LSTM generation
        print("Loading LSTM model...")
        lstm = load_lstm(config.paths.lstm_path, device)

        recommendations = generate_playlist(
            seed_uris=seed_uris,
            lstm=lstm,
            vocab=vocab,
            num_tracks=args.num_tracks,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )

    elif args.mode == "two-stage":
        # Two-stage retrieval + ranking
        print("Loading Item2Vec model...")
        item2vec = load_item2vec(config.paths.item2vec_path, device)

        print("Building retriever...")
        if config.paths.faiss_index_path.exists():
            retriever = CandidateRetriever.load(
                config.paths.faiss_index_path,
                item2vec.get_embeddings(),
                nprobe=config.inference.nprobe,
            )
        else:
            retriever = build_retriever_from_item2vec(
                item2vec, config.inference, config.paths.faiss_index_path
            )

        print("Loading LSTM model...")
        lstm = load_lstm(config.paths.lstm_path, device)
        ranker = LSTMRanker(lstm, device)

        config.inference.num_recommendations = args.num_tracks
        recommendations = two_stage_recommend(
            seed_uris=seed_uris,
            retriever=retriever,
            ranker=ranker,
            vocab=vocab,
            config=config.inference,
        )

    else:
        # Item2Vec only (nearest neighbors)
        print("Loading Item2Vec model...")
        item2vec = load_item2vec(config.paths.item2vec_path, device)

        seed_ids = vocab.encode(seed_uris)
        seed_ids = [idx for idx in seed_ids if idx != vocab.unk_idx]

        # Get centroid of seed tracks and find nearest neighbors
        embeddings = item2vec.get_embeddings()
        retriever = CandidateRetriever(embeddings, nprobe=config.inference.nprobe)
        candidate_ids, scores = retriever.retrieve(seed_ids, k=args.num_tracks)
        recommendations = vocab.decode(candidate_ids.tolist(), skip_special=True)

    # Output recommendations
    print(f"\nGenerated {len(recommendations)} recommendations:")
    print("-" * 50)
    for i, uri in enumerate(recommendations, 1):
        print(f"{i:3d}. {uri}")

    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"seed_tracks": seed_uris, "recommendations": recommendations}, f, indent=2)
        print(f"\nSaved to {args.output}")


def evaluate_command(args):
    """Evaluate model on test set."""
    from .config import Config, get_device
    from .build_vocab import Vocabulary, load_splits
    from .train_item2vec import load_item2vec
    from .train_seq_model import load_lstm
    from .infer_candidates import (
        CandidateRetriever, LSTMRanker, two_stage_recommend,
        compute_metrics, create_test_splits, build_retriever_from_item2vec,
    )
    from .utils_mpd import iter_playlists, get_track_uris_from_playlist
    from tqdm import tqdm
    import numpy as np

    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    device = get_device()
    print(f"Using device: {device}")

    # Load vocab and splits
    print("Loading vocabulary and splits...")
    vocab = Vocabulary.load(config.paths.vocab_path)
    _, _, test_pids = load_splits(
        config.paths.train_pids_path,
        config.paths.val_pids_path,
        config.paths.test_pids_path,
    )

    # Load models
    print("Loading models...")
    item2vec = load_item2vec(config.paths.item2vec_path, device)

    if config.paths.faiss_index_path.exists():
        retriever = CandidateRetriever.load(
            config.paths.faiss_index_path,
            item2vec.get_embeddings(),
            nprobe=config.inference.nprobe,
        )
    else:
        retriever = build_retriever_from_item2vec(
            item2vec, config.inference, config.paths.faiss_index_path
        )

    lstm = load_lstm(config.paths.lstm_path, device)
    ranker = LSTMRanker(lstm, device)

    # Evaluate on test playlists
    print(f"\nEvaluating on {len(test_pids)} test playlists...")
    all_metrics = {"r_precision": [], "ndcg": [], "clicks": []}

    num_evaluated = 0
    for playlist in tqdm(iter_playlists(config.paths.mpd_dir), desc="Evaluating"):
        if playlist.pid not in test_pids:
            continue

        track_uris = get_track_uris_from_playlist(playlist)
        if len(track_uris) < 10:
            continue

        # Create seed/ground truth split
        seed_tracks, ground_truth = create_test_splits(track_uris, seed_ratio=args.seed_ratio)

        # Generate recommendations
        predictions = two_stage_recommend(
            seed_uris=seed_tracks,
            retriever=retriever,
            ranker=ranker,
            vocab=vocab,
            config=config.inference,
        )

        # Compute metrics
        metrics = compute_metrics(predictions, ground_truth)
        for k, v in metrics.items():
            if v != float("inf"):
                all_metrics[k].append(v)

        num_evaluated += 1
        if args.max_playlists and num_evaluated >= args.max_playlists:
            break

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Playlists evaluated: {num_evaluated}")
    print(f"R-Precision:  {np.mean(all_metrics['r_precision']):.4f} (+/- {np.std(all_metrics['r_precision']):.4f})")
    print(f"NDCG:         {np.mean(all_metrics['ndcg']):.4f} (+/- {np.std(all_metrics['ndcg']):.4f})")
    print(f"Song Clicks:  {np.mean(all_metrics['clicks']):.2f} (+/- {np.std(all_metrics['clicks']):.2f})")


def stats_command(args):
    """Compute dataset statistics."""
    from .utils_mpd import compute_dataset_statistics
    import json

    stats = compute_dataset_statistics(args.mpd_dir)

    print("\nMPD Statistics:")
    print("=" * 50)
    print(json.dumps(stats, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nSaved to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Spotify Playlist Recommender",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Build vocabulary command
    vocab_parser = subparsers.add_parser("build-vocab", help="Build vocabulary from MPD")
    vocab_parser.add_argument("--mpd-dir", required=True, help="Path to MPD directory")
    vocab_parser.add_argument("--output", "-o", help="Output vocabulary path")
    vocab_parser.add_argument("--config", "-c", help="Path to config YAML")
    vocab_parser.add_argument("--min-freq", type=int, help="Minimum track frequency")
    vocab_parser.add_argument("--skip-splits", action="store_true", help="Skip creating train/val/test splits")

    # Train Item2Vec command
    i2v_parser = subparsers.add_parser("train-item2vec", help="Train Item2Vec model")
    i2v_parser.add_argument("--config", "-c", help="Path to config YAML")
    i2v_parser.add_argument("--epochs", type=int, help="Number of training epochs")
    i2v_parser.add_argument("--batch-size", type=int, help="Batch size")
    i2v_parser.add_argument("--embedding-dim", type=int, help="Embedding dimension")
    i2v_parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation after training")

    # Train LSTM command
    lstm_parser = subparsers.add_parser("train-lstm", help="Train PlaylistLSTM model")
    lstm_parser.add_argument("--config", "-c", help="Path to config YAML")
    lstm_parser.add_argument("--epochs", type=int, help="Number of training epochs")
    lstm_parser.add_argument("--batch-size", type=int, help="Batch size")
    lstm_parser.add_argument("--hidden-dim", type=int, help="LSTM hidden dimension")
    lstm_parser.add_argument("--no-pretrain", action="store_true", help="Don't initialize from Item2Vec")

    # Recommend command
    rec_parser = subparsers.add_parser("recommend", help="Generate playlist recommendations")
    rec_parser.add_argument("--seeds", required=True, help="Comma-separated seed track URIs")
    rec_parser.add_argument("--num-tracks", type=int, default=50, help="Number of tracks to generate")
    rec_parser.add_argument("--mode", choices=["lstm", "two-stage", "item2vec"], default="two-stage",
                           help="Recommendation mode")
    rec_parser.add_argument("--config", "-c", help="Path to config YAML")
    rec_parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    rec_parser.add_argument("--top-k", type=int, default=50, help="Top-k filtering")
    rec_parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling threshold")
    rec_parser.add_argument("--output", "-o", help="Output JSON path")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate on test set")
    eval_parser.add_argument("--config", "-c", help="Path to config YAML")
    eval_parser.add_argument("--seed-ratio", type=float, default=0.25, help="Ratio of tracks to use as seed")
    eval_parser.add_argument("--max-playlists", type=int, help="Maximum playlists to evaluate")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Compute dataset statistics")
    stats_parser.add_argument("--mpd-dir", required=True, help="Path to MPD directory")
    stats_parser.add_argument("--output", "-o", help="Output JSON path")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Dispatch to appropriate command
    commands = {
        "build-vocab": build_vocab_command,
        "train-item2vec": train_item2vec_command,
        "train-lstm": train_lstm_command,
        "recommend": recommend_command,
        "evaluate": evaluate_command,
        "stats": stats_command,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
