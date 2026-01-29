"""Two-stage inference for playlist recommendations."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .build_vocab import Vocabulary, BOS_IDX
from .config import Config, InferenceConfig, get_device
from .models import Item2Vec, PlaylistLSTM


class CandidateRetriever:
    """
    FAISS-based approximate nearest neighbor retrieval for candidate generation.

    Stage 1 of the two-stage recommendation pipeline.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        nprobe: int = 10,
        use_gpu: bool = False,
    ):
        """
        Args:
            embeddings: Track embeddings [vocab_size, embedding_dim].
            nprobe: Number of clusters to probe for IVF index.
            use_gpu: Whether to use GPU for FAISS (CUDA only).
        """
        import faiss

        self.embeddings = embeddings.astype(np.float32)
        self.vocab_size, self.embedding_dim = embeddings.shape

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self.normalized_embeddings = self.embeddings / norms

        # Build FAISS index
        self.index = self._build_index(nprobe, use_gpu)

    def _build_index(self, nprobe: int, use_gpu: bool):
        """Build FAISS IVF index."""
        import faiss

        # Use IVF for large vocabularies, flat for small
        if self.vocab_size > 10_000:
            # IVF index with inner product (cosine on normalized vectors)
            nlist = min(int(np.sqrt(self.vocab_size)), 4096)
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFFlat(
                quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT
            )

            # Train index
            index.train(self.normalized_embeddings)
            index.nprobe = nprobe
        else:
            # Flat index for small vocabularies
            index = faiss.IndexFlatIP(self.embedding_dim)

        # Add vectors
        index.add(self.normalized_embeddings)

        # GPU support (CUDA only, not MPS)
        if use_gpu and faiss.get_num_gpus() > 0:
            index = faiss.index_cpu_to_all_gpus(index)

        return index

    def retrieve(
        self,
        query_ids: List[int],
        k: int = 500,
        exclude_ids: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve top-k candidates based on query track embeddings.

        Args:
            query_ids: List of query track indices.
            k: Number of candidates to retrieve.
            exclude_ids: Track indices to exclude from results.

        Returns:
            Tuple of (candidate_ids, scores) arrays.
        """
        # Get query embeddings and average them
        query_embeddings = self.normalized_embeddings[query_ids]
        query_centroid = query_embeddings.mean(axis=0, keepdims=True).astype(np.float32)

        # Normalize centroid
        norm = np.linalg.norm(query_centroid)
        if norm > 0:
            query_centroid /= norm

        # Search
        actual_k = k + len(query_ids) + (len(exclude_ids) if exclude_ids else 0)
        actual_k = min(actual_k, self.vocab_size)

        scores, indices = self.index.search(query_centroid, actual_k)
        scores = scores[0]
        indices = indices[0]

        # Filter out query tracks and excluded tracks
        exclude_set = set(query_ids)
        if exclude_ids:
            exclude_set.update(exclude_ids)
        # Also exclude special tokens
        exclude_set.update([0, 1, 2, 3])

        filtered_results = [
            (idx, score) for idx, score in zip(indices, scores)
            if idx not in exclude_set and idx >= 0
        ]

        # Return top-k after filtering
        filtered_results = filtered_results[:k]
        if filtered_results:
            candidate_ids = np.array([r[0] for r in filtered_results])
            scores = np.array([r[1] for r in filtered_results])
        else:
            candidate_ids = np.array([], dtype=np.int64)
            scores = np.array([], dtype=np.float32)

        return candidate_ids, scores

    def save(self, path: Path | str) -> None:
        """Save FAISS index to file."""
        import faiss
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path | str, embeddings: np.ndarray, nprobe: int = 10) -> "CandidateRetriever":
        """Load FAISS index from file."""
        import faiss

        obj = cls.__new__(cls)
        obj.embeddings = embeddings.astype(np.float32)
        obj.vocab_size, obj.embedding_dim = embeddings.shape

        norms = np.linalg.norm(obj.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        obj.normalized_embeddings = obj.embeddings / norms

        obj.index = faiss.read_index(str(path))
        if hasattr(obj.index, 'nprobe'):
            obj.index.nprobe = nprobe

        return obj


class LSTMRanker:
    """
    LSTM-based ranker for reranking candidate tracks.

    Stage 2 of the two-stage recommendation pipeline.
    """

    def __init__(self, model: PlaylistLSTM, device: Optional[torch.device] = None):
        """
        Args:
            model: Trained PlaylistLSTM model.
            device: Device for inference.
        """
        self.model = model
        self.device = device or get_device()
        self.model = self.model.to(self.device)
        self.model.eval()

    def rank(
        self,
        seed_ids: List[int],
        candidate_ids: List[int],
        batch_size: int = 256,
    ) -> List[Tuple[int, float]]:
        """
        Rank candidates by P(candidate | seed_sequence).

        Args:
            seed_ids: Seed track indices.
            candidate_ids: Candidate track indices to rank.

        Returns:
            List of (track_id, score) tuples sorted by score descending.
        """
        if not candidate_ids:
            return []

        # Create seed sequence with BOS
        seed_tensor = torch.tensor([BOS_IDX] + list(seed_ids), dtype=torch.long)
        seed_tensor = seed_tensor.unsqueeze(0).to(self.device)

        scores = []

        with torch.no_grad():
            # Get LSTM hidden state after processing seed
            _, hidden = self.model(seed_tensor)

            # Score each candidate in batches
            for i in range(0, len(candidate_ids), batch_size):
                batch_candidates = candidate_ids[i:i + batch_size]
                batch_size_actual = len(batch_candidates)

                # Expand hidden state for batch
                h, c = hidden
                h_batch = h.expand(-1, batch_size_actual, -1).contiguous()
                c_batch = c.expand(-1, batch_size_actual, -1).contiguous()

                # Last input is the last seed token
                last_input = seed_tensor[:, -1:].expand(batch_size_actual, -1)

                # Get logits for next token prediction
                logits, _ = self.model(last_input, hidden=(h_batch, c_batch))
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

                # Get scores for candidates
                candidate_tensor = torch.tensor(batch_candidates, dtype=torch.long, device=self.device)
                batch_scores = log_probs[torch.arange(batch_size_actual), candidate_tensor]
                scores.extend(batch_scores.cpu().tolist())

        # Sort by score descending
        ranked = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked


def two_stage_recommend(
    seed_uris: List[str],
    retriever: CandidateRetriever,
    ranker: LSTMRanker,
    vocab: Vocabulary,
    config: InferenceConfig,
) -> List[str]:
    """
    Generate recommendations using two-stage retrieval + ranking.

    Args:
        seed_uris: Seed track URIs.
        retriever: CandidateRetriever for stage 1.
        ranker: LSTMRanker for stage 2.
        vocab: Vocabulary for encoding/decoding.
        config: Inference configuration.

    Returns:
        List of recommended track URIs.
    """
    # Encode seed tracks
    seed_ids = vocab.encode(seed_uris)
    seed_ids = [idx for idx in seed_ids if idx != vocab.unk_idx]

    if not seed_ids:
        return []

    # Stage 1: Retrieve candidates
    candidate_ids, _ = retriever.retrieve(
        query_ids=seed_ids,
        k=config.num_candidates,
    )

    if len(candidate_ids) == 0:
        return []

    # Stage 2: Rank candidates with LSTM
    ranked = ranker.rank(seed_ids, candidate_ids.tolist())

    # Return top recommendations
    top_ids = [idx for idx, _ in ranked[:config.num_recommendations]]
    return vocab.decode(top_ids, skip_special=True)


def build_retriever_from_item2vec(
    item2vec: Item2Vec,
    config: InferenceConfig,
    save_path: Optional[Path] = None,
) -> CandidateRetriever:
    """
    Build a CandidateRetriever from a trained Item2Vec model.

    Args:
        item2vec: Trained Item2Vec model.
        config: Inference configuration.
        save_path: Optional path to save the FAISS index.

    Returns:
        CandidateRetriever instance.
    """
    embeddings = item2vec.get_embeddings()
    retriever = CandidateRetriever(embeddings, nprobe=config.nprobe)

    if save_path:
        retriever.save(save_path)

    return retriever


def compute_metrics(
    predictions: List[str],
    ground_truth: List[str],
) -> dict:
    """
    Compute recommendation metrics.

    Args:
        predictions: List of predicted track URIs.
        ground_truth: List of ground truth track URIs.

    Returns:
        Dictionary of metrics.
    """
    gt_set = set(ground_truth)
    n_gt = len(gt_set)

    if n_gt == 0:
        return {"r_precision": 0.0, "ndcg": 0.0, "clicks": float("inf")}

    # R-Precision: precision at R where R = len(ground_truth)
    r = min(n_gt, len(predictions))
    r_hits = sum(1 for p in predictions[:r] if p in gt_set)
    r_precision = r_hits / r if r > 0 else 0.0

    # NDCG
    dcg = 0.0
    for i, p in enumerate(predictions):
        if p in gt_set:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because ranks start at 1

    # Ideal DCG
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(n_gt, len(predictions))))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    # Song Clicks: position of first relevant track / 10
    clicks = float("inf")
    for i, p in enumerate(predictions):
        if p in gt_set:
            clicks = i // 10
            break

    return {
        "r_precision": r_precision,
        "ndcg": ndcg,
        "clicks": clicks,
    }


def create_test_splits(
    playlist_tracks: List[str],
    seed_ratio: float = 0.25,
) -> Tuple[List[str], List[str]]:
    """
    Create train/test split for evaluation.

    Args:
        playlist_tracks: Full playlist track URIs.
        seed_ratio: Ratio of tracks to use as seed.

    Returns:
        Tuple of (seed_tracks, ground_truth_tracks).
    """
    n_seed = max(1, int(len(playlist_tracks) * seed_ratio))
    return playlist_tracks[:n_seed], playlist_tracks[n_seed:]
