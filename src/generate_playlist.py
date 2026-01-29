"""Playlist generation using trained models."""

from typing import List, Optional, Set

import torch
import torch.nn.functional as F

from .build_vocab import Vocabulary, BOS_IDX, EOS_IDX
from .config import InferenceConfig, get_device
from .infer_candidates import CandidateRetriever, LSTMRanker
from .models import PlaylistLSTM


def generate_playlist(
    seed_uris: List[str],
    lstm: PlaylistLSTM,
    vocab: Vocabulary,
    num_tracks: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    diversity_penalty: float = 0.0,
    device: Optional[torch.device] = None,
) -> List[str]:
    """
    Generate playlist continuation using autoregressive LSTM sampling.

    Args:
        seed_uris: Seed track URIs.
        lstm: Trained PlaylistLSTM model.
        vocab: Vocabulary for encoding/decoding.
        num_tracks: Number of tracks to generate.
        temperature: Sampling temperature (higher = more random).
        top_k: Top-k filtering parameter.
        top_p: Nucleus sampling probability threshold.
        diversity_penalty: Penalty for repeating tracks.
        device: Device for inference.

    Returns:
        List of generated track URIs (not including seed).
    """
    if device is None:
        device = get_device()

    lstm = lstm.to(device)
    lstm.eval()

    # Encode seed tracks
    seed_ids = vocab.encode(seed_uris)
    seed_ids = [idx for idx in seed_ids if idx != vocab.unk_idx]

    if not seed_ids:
        return []

    # Prepare seed sequence with BOS
    seed_tensor = torch.tensor([BOS_IDX] + seed_ids, dtype=torch.long, device=device)

    # Track generated tokens to apply diversity penalty
    generated_set: Set[int] = set(seed_ids)
    generated_ids: List[int] = []

    with torch.no_grad():
        # Process seed sequence
        logits, hidden = lstm(seed_tensor.unsqueeze(0))

        for _ in range(num_tracks):
            # Get logits for next token
            next_logits = logits[0, -1, :].clone()

            # Apply diversity penalty to already generated tracks
            if diversity_penalty > 0:
                for idx in generated_set:
                    next_logits[idx] -= diversity_penalty

            # Apply temperature
            next_logits = next_logits / temperature

            # Mask special tokens
            next_logits[0] = float("-inf")  # PAD
            next_logits[1] = float("-inf")  # UNK
            next_logits[2] = float("-inf")  # BOS
            # Allow EOS to be sampled but with low probability initially

            # Top-k filtering
            if top_k > 0:
                threshold = torch.topk(next_logits, top_k)[0][-1]
                next_logits[next_logits < threshold] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Find cutoff
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Stop if EOS
            if next_token == EOS_IDX:
                break

            generated_ids.append(next_token)
            generated_set.add(next_token)

            # Prepare next input
            next_input = torch.tensor([[next_token]], dtype=torch.long, device=device)
            logits, hidden = lstm(next_input, hidden=hidden)

    return vocab.decode(generated_ids, skip_special=True)


def generate_with_two_stage(
    seed_uris: List[str],
    retriever: CandidateRetriever,
    ranker: LSTMRanker,
    vocab: Vocabulary,
    num_tracks: int = 50,
    candidates_per_step: int = 100,
    top_k_select: int = 5,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
) -> List[str]:
    """
    Generate playlist using iterative two-stage retrieval + ranking.

    This method generates tracks one at a time, each time:
    1. Retrieving candidates similar to current playlist
    2. Ranking candidates with LSTM
    3. Sampling from top-ranked candidates

    Args:
        seed_uris: Seed track URIs.
        retriever: CandidateRetriever for candidate generation.
        ranker: LSTMRanker for ranking.
        vocab: Vocabulary for encoding/decoding.
        num_tracks: Number of tracks to generate.
        candidates_per_step: Number of candidates to retrieve at each step.
        top_k_select: Number of top candidates to sample from.
        temperature: Sampling temperature.
        device: Device for inference.

    Returns:
        List of generated track URIs (not including seed).
    """
    if device is None:
        device = get_device()

    # Encode seed tracks
    current_ids = vocab.encode(seed_uris)
    current_ids = [idx for idx in current_ids if idx != vocab.unk_idx]

    if not current_ids:
        return []

    generated_ids: List[int] = []
    generated_set: Set[int] = set(current_ids)

    for _ in range(num_tracks):
        # Stage 1: Retrieve candidates based on current playlist
        candidate_ids, _ = retriever.retrieve(
            query_ids=current_ids + generated_ids,
            k=candidates_per_step,
            exclude_ids=list(generated_set),
        )

        if len(candidate_ids) == 0:
            break

        # Stage 2: Rank candidates with LSTM
        ranked = ranker.rank(current_ids + generated_ids, candidate_ids.tolist())

        if not ranked:
            break

        # Sample from top-k candidates
        top_candidates = ranked[:top_k_select]
        scores = torch.tensor([score for _, score in top_candidates])

        # Apply temperature and softmax
        probs = F.softmax(scores / temperature, dim=-1)
        selected_idx = torch.multinomial(probs, num_samples=1).item()
        selected_track = top_candidates[selected_idx][0]

        generated_ids.append(selected_track)
        generated_set.add(selected_track)

    return vocab.decode(generated_ids, skip_special=True)


def generate_diverse_playlist(
    seed_uris: List[str],
    lstm: PlaylistLSTM,
    vocab: Vocabulary,
    num_tracks: int = 50,
    num_samples: int = 5,
    temperature: float = 1.2,
    top_p: float = 0.95,
    device: Optional[torch.device] = None,
) -> List[List[str]]:
    """
    Generate multiple diverse playlist continuations.

    Useful for giving users options to choose from.

    Args:
        seed_uris: Seed track URIs.
        lstm: Trained PlaylistLSTM model.
        vocab: Vocabulary for encoding/decoding.
        num_tracks: Number of tracks to generate per playlist.
        num_samples: Number of different playlists to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        device: Device for inference.

    Returns:
        List of generated playlists (each is a list of track URIs).
    """
    playlists = []
    for _ in range(num_samples):
        playlist = generate_playlist(
            seed_uris=seed_uris,
            lstm=lstm,
            vocab=vocab,
            num_tracks=num_tracks,
            temperature=temperature,
            top_p=top_p,
            diversity_penalty=0.5,  # Encourage diversity within each playlist
            device=device,
        )
        playlists.append(playlist)

    return playlists


def extend_playlist_interactively(
    current_uris: List[str],
    lstm: PlaylistLSTM,
    vocab: Vocabulary,
    num_suggestions: int = 10,
    temperature: float = 0.8,
    top_k: int = 50,
    device: Optional[torch.device] = None,
) -> List[tuple]:
    """
    Get ranked suggestions for the next track.

    Useful for interactive playlist building where user selects each track.

    Args:
        current_uris: Current playlist track URIs.
        lstm: Trained PlaylistLSTM model.
        vocab: Vocabulary for encoding/decoding.
        num_suggestions: Number of suggestions to return.
        temperature: Sampling temperature.
        top_k: Number of top candidates to consider.
        device: Device for inference.

    Returns:
        List of (track_uri, probability) tuples sorted by probability.
    """
    if device is None:
        device = get_device()

    lstm = lstm.to(device)
    lstm.eval()

    # Encode current playlist
    current_ids = vocab.encode(current_uris)
    current_ids = [idx for idx in current_ids if idx != vocab.unk_idx]

    if not current_ids:
        return []

    # Prepare sequence with BOS
    input_tensor = torch.tensor([BOS_IDX] + current_ids, dtype=torch.long, device=device)

    with torch.no_grad():
        logits, _ = lstm(input_tensor.unsqueeze(0))
        next_logits = logits[0, -1, :]

        # Apply temperature
        next_logits = next_logits / temperature

        # Mask special tokens and already included tracks
        next_logits[0] = float("-inf")  # PAD
        next_logits[1] = float("-inf")  # UNK
        next_logits[2] = float("-inf")  # BOS
        next_logits[3] = float("-inf")  # EOS
        for idx in current_ids:
            next_logits[idx] = float("-inf")

        # Get probabilities
        probs = F.softmax(next_logits, dim=-1)

        # Get top-k
        top_probs, top_indices = torch.topk(probs, min(num_suggestions, top_k))

    suggestions = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        uri = vocab.idx2track.get(idx, None)
        if uri and not uri.startswith("<"):
            suggestions.append((uri, prob))

    return suggestions[:num_suggestions]
