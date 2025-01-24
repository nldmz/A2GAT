# model.py
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, Optional


class BipartiteGraphEmbedding(nn.Module):
    """Base class for bipartite graph embedding models."""

    def __init__(self):
        super(BipartiteGraphEmbedding, self).__init__()

    def get_user_embedding(self, user_indices: torch.Tensor) -> torch.Tensor:
        """Get user embeddings for given indices."""
        raise NotImplementedError

    def get_item_embedding(self) -> torch.Tensor:
        """Get all item embeddings."""
        raise NotImplementedError


class AdaptiveAnchorGAT(BipartiteGraphEmbedding):
    """Adaptive Anchor-based Graph Attention Networks (A2GAT) for bipartite graphs."""

    def __init__(
            self,
            embedding_dim: int,
            num_users: int,
            num_items: int,
            num_anchors: int,
            anchor_dim: int,
            adjacency_matrix: torch.Tensor
    ):
        """Initialize A2GAT model.

        Args:
            embedding_dim: Dimension of embeddings
            num_users: Number of users
            num_items: Number of items 
            num_anchors: Number of anchor points
            anchor_dim: Dimension of anchor embeddings
            adjacency_matrix: Adjacency matrix of the bipartite graph
        """
        super(AdaptiveAnchorGAT, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_users = num_users
        self.num_items = num_items

        # Initialize node embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self._init_embeddings()

        # Initialize network components
        self.adaptive_anchor_conv = AdaptiveAnchorConvolution(
            embedding_dim,
            num_anchors,
            anchor_dim
        )
        self.adjacency_matrix = adjacency_matrix
        self.bipartite_gat = BipartiteGATLayer(
            in_features=embedding_dim,
            out_features=embedding_dim,
            dropout=0.2,
            alpha=0.2
        )

    def _init_embeddings(self):
        """Initialize embedding weights using Xavier normal initialization."""
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

    def forward(
            self,
            user_indices: torch.Tensor,
            pos_item_indices: torch.Tensor,
            neg_item_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of A2GAT.

        Args:
            user_indices: User indices
            pos_item_indices: Positive item indices
            neg_item_indices: Negative item indices (optional)
        """
        user_emb = self.user_embedding(user_indices)
        pos_item_emb = self.item_embedding(pos_item_indices)

        # Apply anchor-based convolution
        user_emb = self.adaptive_anchor_conv(user_emb)

        if neg_item_indices is None:
            # Full cross entropy mode
            item_scores = self.item_embedding.weight
            predictions = user_emb @ item_scores.transpose(0, 1)
        else:
            # Mini batch cross entropy mode
            neg_item_emb = self.item_embedding(neg_item_indices)
            item_scores = torch.cat(
                (pos_item_emb.unsqueeze(dim=1), neg_item_emb),
                dim=1
            )
            predictions = (user_emb.unsqueeze(dim=1) * item_scores).sum(-1)

        return predictions, (user_emb, pos_item_emb)

    def get_user_embedding(self, user_indices: torch.Tensor) -> torch.Tensor:
        """Get transformed user embeddings."""
        user_emb = self.user_embedding(user_indices)
        return self.adaptive_anchor_conv(user_emb)

    def get_item_embedding(self) -> torch.Tensor:
        """Get all item embeddings."""
        return self.item_embedding.weight


class FullyConnectedAttention(nn.Module):
    """Fully Connected Attention (FCA) mechanism for global information exchange."""

    def __init__(self, input_dim: int, output_dim: int):
        super(FullyConnectedAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialize learnable parameters
        self.weight_matrix = nn.Parameter(
            torch.zeros(size=(input_dim, output_dim))
        )
        self.attention_vector = nn.Parameter(
            torch.zeros(size=(2 * output_dim, 1))
        )
        self._init_attention_weights()

    def _init_attention_weights(self):
        """Initialize attention weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight_matrix.data, gain=1.414)
        nn.init.xavier_uniform_(self.attention_vector.data, gain=1.414)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """Apply fully connected attention mechanism."""
        num_nodes = node_features.size(0)

        # Transform features
        transformed_features = torch.mm(node_features, self.weight_matrix)

        # Compute attention scores
        attention_input = torch.cat((
            transformed_features.unsqueeze(1).repeat(1, num_nodes, 1),
            transformed_features.unsqueeze(0).repeat(num_nodes, 1, 1)
        ), dim=2)

        attention_scores = torch.matmul(
            attention_input,
            self.attention_vector
        ).squeeze(2)

        # Apply attention weights
        attention_weights = F.softmax(attention_scores, dim=1)
        attended_features = torch.matmul(
            attention_weights,
            transformed_features
        )

        return attended_features


class AdaptiveAnchorConvolution(nn.Module):
    """Adaptive Anchor-based Convolution with attention mechanisms."""

    def __init__(self, feature_dim: int, num_anchors: int, anchor_dim: int):
        super(AdaptiveAnchorConvolution, self).__init__()
        self.feature_dim = feature_dim
        self.num_anchors = num_anchors
        self.anchor_dim = anchor_dim

        # Initialize components
        self.anchor_points = AdaptiveAnchors(num_anchors, anchor_dim)
        self.receive_attention = FullyConnectedAttention(num_anchors, feature_dim)
        self.send_attention = FullyConnectedAttention(feature_dim, anchor_dim)

        # Layer normalization
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.anchor_norm = nn.LayerNorm(num_anchors)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass of adaptive anchor convolution."""
        # Normalize input features
        normalized_features = self.feature_norm(features)

        # Apply send attention and anchor projection
        sent_features = self.send_attention(normalized_features)
        anchor_projected = self.anchor_points(sent_features)

        # Normalize and receive attention
        normalized_anchors = self.anchor_norm(anchor_projected)
        received_features = self.receive_attention(normalized_anchors)

        # Non-linear transformation and residual connection
        transformed_features = torch.sin(received_features)
        return features + transformed_features


class AdaptiveAnchors(nn.Module):
    """Adaptive anchor points for global information aggregation."""

    def __init__(self, num_anchors: int, anchor_dim: int):
        super(AdaptiveAnchors, self).__init__()
        self.anchor_embeddings = nn.Parameter(
            torch.empty(num_anchors, anchor_dim)
        )
        self._init_anchor_embeddings()

    def _init_anchor_embeddings(self):
        """Initialize anchor embeddings using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.anchor_embeddings)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Project features onto anchor space."""
        return features @ self.anchor_embeddings.transpose(0, 1)


class BipartiteGATLayer(nn.Module):
    """Bipartite Graph Attention Layer with entropy regularization."""

    def __init__(
            self,
            in_features: int,
            out_features: int,
            dropout: float,
            alpha: float,
            concat: bool = True
    ):
        super(BipartiteGATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # Initialize attention parameters
        self.weight_matrix = nn.Parameter(
            torch.zeros(size=(in_features, out_features))
        )
        self.attention_vector = nn.Parameter(
            torch.zeros(size=(2 * out_features, 1))
        )
        self._init_attention_weights()

        self.leaky_relu = nn.LeakyReLU(self.alpha)

    def _init_attention_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight_matrix.data, gain=1.414)
        nn.init.xavier_uniform_(self.attention_vector.data, gain=1.414)

    def forward(
            self,
            node_features: torch.Tensor,
            adjacency_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the BipartiteGAT layer."""
        # Transform features
        transformed_features = torch.mm(node_features, self.weight_matrix)

        # Compute attention
        attention_input = self._prepare_attention_input(transformed_features)
        attention_scores = torch.matmul(
            attention_input,
            self.attention_vector
        ).squeeze(2)

        # Apply attention mask and weights
        attention_mask = -9e15 * torch.ones_like(attention_scores)
        masked_attention = torch.where(
            adjacency_matrix > 0,
            attention_scores,
            attention_mask
        )
        attention_weights = F.softmax(masked_attention, dim=1)
        attention_weights = F.dropout(
            attention_weights,
            self.dropout,
            training=self.training
        )

        # Aggregate features
        output_features = torch.matmul(attention_weights, transformed_features)

        if self.concat:
            return F.elu(output_features)
        return output_features

    def _prepare_attention_input(
            self,
            transformed_features: torch.Tensor
    ) -> torch.Tensor:
        """Prepare input for attention mechanism."""
        num_nodes = transformed_features.size(0)
        repeated_features = transformed_features.repeat_interleave(num_nodes, dim=0)
        alternating_features = transformed_features.repeat(num_nodes, 1)

        combined_features = torch.cat([
            repeated_features,
            alternating_features
        ], dim=1)

        return combined_features.view(
            num_nodes,
            num_nodes,
            2 * self.out_features
        )


class L2Regularizer(nn.Module):
    """L2 regularization  with entropy regularization for attention weights."""

    def __init__(self, weight: float):
        super(L2Regularizer, self).__init__()
        self.weight = weight

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply L2 regularization with entropy regularization."""
        l2_norm = sum(torch.norm(emb, p=2) for emb in embeddings)
        return self.weight * l2_norm