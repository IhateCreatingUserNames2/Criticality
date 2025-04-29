"""
Embedding Utils - Adaptive Cross-Model Similarity for Different Embedding Dimensions

This module provides utilities for computing similarity between embeddings
from different models with potentially different dimensions.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_adaptive_similarity(embedding1, embedding2, memory_type1=None, memory_type2=None):
    """
    Compute similarity between embeddings with potentially different dimensions

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        memory_type1: Memory type of first embedding (optional)
        memory_type2: Memory type of second embedding (optional)

    Returns:
        Similarity score (0-1)
    """
    if embedding1 is None or embedding2 is None:
        return 0.5

    # If dimensions match, use direct comparison
    if embedding1.shape == embedding2.shape:
        return float(cosine_similarity([embedding1], [embedding2])[0][0])

    # For different dimensions, use adaptive comparison
    return safe_cross_dimension_similarity(embedding1, embedding2)


def safe_cross_dimension_similarity(embedding1, embedding2):
    """
    Safely compare embeddings with different dimensions

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Similarity score (0-1)
    """
    try:
        # Get minimum dimension to avoid index errors
        min_dim = min(embedding1.shape[0], embedding2.shape[0])

        # Use the aligned dimensions for comparison
        truncated_emb1 = embedding1[:min_dim]
        truncated_emb2 = embedding2[:min_dim]

        # Compute similarity on truncated vectors
        similarity = float(cosine_similarity([truncated_emb1], [truncated_emb2])[0][0])

        # Apply a slight penalty for dimension mismatch
        # (This encourages using same-dimension comparisons when possible)
        dim_difference_ratio = abs(embedding1.shape[0] - embedding2.shape[0]) / max(embedding1.shape[0],
                                                                                    embedding2.shape[0])
        similarity_penalty = 0.1 * dim_difference_ratio

        return max(0.0, similarity - similarity_penalty)
    except Exception as e:
        print(f"[Error] Cross-dimension similarity calculation failed: {e}")
        # Return a neutral similarity
        return 0.5


def memory_similarity(memory1, memory2, memory_blossom):
    """
    Compute similarity between two memory objects

    Args:
        memory1: First memory object
        memory2: Second memory object
        memory_blossom: MemoryBlossom instance

    Returns:
        Similarity score (0-1)
    """
    # If same memory type, use direct comparison
    if memory1.memory_type == memory2.memory_type:
        return compute_adaptive_similarity(memory1.embedding, memory2.embedding)

    # For different memory types, compute using type-aware approach
    return compute_memory_cross_type_similarity(memory1, memory2, memory_blossom)


def compute_memory_cross_type_similarity(memory1, memory2, memory_blossom):
    """
    Compute similarity between memories of different types

    Args:
        memory1: First memory object
        memory2: Second memory object
        memory_blossom: MemoryBlossom instance

    Returns:
        Similarity score (0-1)
    """
    try:
        # Use the content as a bridge between embedding spaces
        # Re-encode memory1's content with memory2's model
        model2 = memory_blossom.embedding_models.get(memory2.memory_type)
        if model2 is None:
            return compute_adaptive_similarity(memory1.embedding, memory2.embedding)

        bridge_embedding = model2.encode([memory1.content])[0]

        # Calculate similarity between bridge embedding and memory2's embedding
        bridge_similarity = compute_adaptive_similarity(bridge_embedding, memory2.embedding)

        # Also calculate direct similarity as a fallback
        direct_similarity = compute_adaptive_similarity(memory1.embedding, memory2.embedding)

        # Weighted combination (favoring bridge similarity)
        return bridge_similarity * 0.7 + direct_similarity * 0.3
    except Exception as e:
        print(f"[Error] Cross-type similarity calculation failed: {e}")
        # Fall back to direct comparison
        return compute_adaptive_similarity(memory1.embedding, memory2.embedding)