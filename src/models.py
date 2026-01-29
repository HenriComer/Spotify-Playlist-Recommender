"""Neural network models for Spotify Playlist Recommender."""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Item2Vec(nn.Module):
    """
    Skip-gram model with negative sampling for learning track embeddings.

    Uses separate embedding matrices for center and context words,
    similar to Word2Vec implementation.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        padding_idx: int = 0,
    ):
        """
        Args:
            vocab_size: Size of the vocabulary.
            embedding_dim: Dimension of embeddings.
            padding_idx: Index of padding token.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Center word embeddings (used for final track representations)
        self.center_embeddings = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx, sparse=True
        )

        # Context word embeddings (used for prediction)
        self.context_embeddings = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx, sparse=True
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings with uniform distribution."""
        init_range = 0.5 / self.embedding_dim
        self.center_embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)
        # Zero out padding
        self.center_embeddings.weight.data[0].zero_()
        self.context_embeddings.weight.data[0].zero_()

    def forward(
        self,
        center_ids: torch.Tensor,
        context_ids: torch.Tensor,
        negative_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute negative sampling loss.

        Args:
            center_ids: Center word indices [batch_size].
            context_ids: Positive context indices [batch_size].
            negative_ids: Negative sample indices [batch_size, num_negatives].

        Returns:
            Scalar loss tensor.
        """
        # Get embeddings
        center_embeds = self.center_embeddings(center_ids)  # [B, D]
        context_embeds = self.context_embeddings(context_ids)  # [B, D]
        neg_embeds = self.context_embeddings(negative_ids)  # [B, K, D]

        # Positive score: center · context
        pos_score = torch.sum(center_embeds * context_embeds, dim=1)  # [B]
        pos_loss = F.logsigmoid(pos_score)

        # Negative score: center · negative
        # [B, 1, D] @ [B, D, K] -> [B, 1, K] -> [B, K]
        neg_score = torch.bmm(
            center_embeds.unsqueeze(1),
            neg_embeds.transpose(1, 2)
        ).squeeze(1)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)  # [B]

        # Total loss (negative because we maximize log likelihood)
        loss = -(pos_loss + neg_loss).mean()

        return loss

    def get_embeddings(self) -> np.ndarray:
        """Get the center embeddings as numpy array."""
        return self.center_embeddings.weight.detach().cpu().numpy()

    def get_similar(
        self,
        track_idx: int,
        k: int = 10,
        exclude_special: bool = True,
    ) -> List[Tuple[int, float]]:
        """
        Find most similar tracks to a given track.

        Args:
            track_idx: Index of query track.
            k: Number of similar tracks to return.
            exclude_special: Whether to exclude special tokens from results.

        Returns:
            List of (track_idx, similarity_score) tuples.
        """
        embeddings = self.get_embeddings()

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        normalized = embeddings / norms

        # Compute cosine similarity
        query = normalized[track_idx]
        similarities = np.dot(normalized, query)

        # Get top-k (excluding self)
        if exclude_special:
            # Set similarity to -inf for special tokens (0-3)
            similarities[:4] = -np.inf
        similarities[track_idx] = -np.inf

        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [(int(idx), float(similarities[idx])) for idx in top_k_indices]


class PlaylistLSTM(nn.Module):
    """
    Multi-layer LSTM for playlist sequence prediction.

    Supports optional weight tying and initialization from Item2Vec embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0,
        tie_weights: bool = True,
    ):
        """
        Args:
            vocab_size: Size of the vocabulary.
            embedding_dim: Dimension of embeddings.
            hidden_dim: LSTM hidden dimension.
            num_layers: Number of LSTM layers.
            dropout: Dropout probability.
            padding_idx: Index of padding token.
            tie_weights: Whether to tie input and output embeddings.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.padding_idx = padding_idx
        self.tie_weights = tie_weights

        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output projection
        if tie_weights:
            # Project hidden to embedding dim, then use embedding weights
            self.output_proj = nn.Linear(hidden_dim, embedding_dim)
            self.output = nn.Linear(embedding_dim, vocab_size, bias=False)
            self.output.weight = self.embedding.weight
        else:
            self.output_proj = nn.Identity()
            self.output = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Initialize embeddings
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.embedding.weight.data[self.padding_idx].zero_()

        # Initialize LSTM
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def init_from_item2vec(self, item2vec: Item2Vec) -> None:
        """
        Initialize embeddings from a trained Item2Vec model.

        Args:
            item2vec: Trained Item2Vec model.
        """
        with torch.no_grad():
            pretrained = item2vec.center_embeddings.weight.data
            if pretrained.shape == self.embedding.weight.shape:
                self.embedding.weight.copy_(pretrained)
                print("Initialized embeddings from Item2Vec")
            else:
                print(f"Shape mismatch: {pretrained.shape} vs {self.embedding.weight.shape}")

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for sequence prediction.

        Args:
            input_ids: Input token indices [batch_size, seq_len].
            lengths: Sequence lengths for packing [batch_size].
            hidden: Initial hidden state (h, c) tuple.

        Returns:
            Tuple of (logits, hidden_state).
            logits: [batch_size, seq_len, vocab_size]
            hidden_state: (h, c) tuple for next step
        """
        batch_size, seq_len = input_ids.shape

        # Embed input
        embeds = self.embedding(input_ids)  # [B, L, E]
        embeds = self.dropout(embeds)

        # Pack sequences if lengths provided
        if lengths is not None:
            # Sort by length (required for pack_padded_sequence)
            lengths_sorted, sort_idx = lengths.sort(descending=True)
            embeds_sorted = embeds[sort_idx]

            packed = nn.utils.rnn.pack_padded_sequence(
                embeds_sorted, lengths_sorted.cpu(), batch_first=True
            )
            lstm_out, hidden = self.lstm(packed, hidden)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=seq_len
            )

            # Unsort
            _, unsort_idx = sort_idx.sort()
            lstm_out = lstm_out[unsort_idx]
        else:
            lstm_out, hidden = self.lstm(embeds, hidden)

        # Project to vocab
        lstm_out = self.dropout(lstm_out)
        proj = self.output_proj(lstm_out)
        logits = self.output(proj)

        return logits, hidden

    def generate(
        self,
        seed_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_idx: int = 3,
    ) -> List[int]:
        """
        Generate playlist continuation using nucleus sampling.

        Args:
            seed_ids: Seed track indices [seq_len].
            max_length: Maximum number of tracks to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering parameter.
            top_p: Nucleus sampling probability threshold.
            eos_idx: End of sequence token index.

        Returns:
            List of generated track indices.
        """
        self.eval()
        device = next(self.parameters()).device

        # Add batch dimension
        if seed_ids.dim() == 1:
            seed_ids = seed_ids.unsqueeze(0)
        seed_ids = seed_ids.to(device)

        generated = []
        hidden = None

        # Process seed sequence
        with torch.no_grad():
            logits, hidden = self.forward(seed_ids, hidden=hidden)
            next_logits = logits[0, -1, :]  # Last position

            for _ in range(max_length):
                # Apply temperature
                next_logits = next_logits / temperature

                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = float("-inf")

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift to keep first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_logits[indices_to_remove] = float("-inf")

                # Sample
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                if next_token == eos_idx:
                    break

                generated.append(next_token)

                # Prepare next input
                next_input = torch.tensor([[next_token]], device=device)
                logits, hidden = self.forward(next_input, hidden=hidden)
                next_logits = logits[0, -1, :]

        return generated

    def score_sequences(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute log probability of target sequences given inputs.

        Args:
            input_ids: Input sequences [batch_size, seq_len].
            target_ids: Target sequences [batch_size, seq_len].
            lengths: Sequence lengths [batch_size].

        Returns:
            Log probabilities [batch_size].
        """
        logits, _ = self.forward(input_ids, lengths)
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather target log probs
        target_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

        # Mask padding
        if lengths is not None:
            mask = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
            mask = mask < lengths.unsqueeze(1)
            target_log_probs = target_log_probs * mask.float()

        return target_log_probs.sum(dim=1)
