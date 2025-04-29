"""
Memory Connector - Advanced Memory Integration System

This module implements a general-purpose memory connection system that discovers
and utilizes relationships between memories without hardcoding specific types.
It provides a more holistic approach to memory connections, treating the memory
system as a complex network existing at the edge of chaos.
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Set, Tuple
from sklearn.metrics.pairwise import cosine_similarity


class MemoryConnector:
    """
    A general-purpose memory connection system that discovers and utilizes
    relationships between memories without hardcoding specific types.

    This system implements principles from complex systems theory, particularly
    the concept that optimal intelligence emerges at the boundary between
    order and chaos - the "critical zone".
    """

    def __init__(self, memory_blossom):
        """Initialize the memory connector with a reference to the MemoryBlossom system."""
        self.memory_blossom = memory_blossom
        self.connection_graph = defaultdict(list)  # memory_id -> [(memory_id, similarity, relation_type)]
        self.memory_clusters = []  # groups of related memories
        self.semantic_fields = {}  # embedding space regions with similar conceptual meaning

    def analyze_all_memories(self):
        """
        Analyze all memories to find connections, clusters, and semantic fields.
        This creates a complex network of memory relationships that can be traversed
        during retrieval.
        """
        all_memories = []

        # Collect all memories across types
        for mem_type in self.memory_blossom.memory_stores:
            all_memories.extend(self.memory_blossom.memory_stores[mem_type])

        if not all_memories:
            print("[MemoryConnector] No memories to analyze")
            return

        print(f"[MemoryConnector] Analyzing connections between {len(all_memories)} memories")

        # Build connection graph
        for i, mem1 in enumerate(all_memories):
            for j, mem2 in enumerate(all_memories[i + 1:], i + 1):
                # Skip self-connections
                if mem1.id == mem2.id:
                    continue

                # Use multiple embedding models to compare memories
                similarities = []
                relation_types = []

                # Try different embedding models for more robust similarity assessment
                for model_name, model in self.memory_blossom.embedding_models.items():
                    try:
                        # Import our similarity function
                        from embedding_utils import compute_adaptive_similarity

                        # Calculate similarity between embeddings using our adaptive function
                        sim = compute_adaptive_similarity(mem1.embedding, mem2.embedding)

                        if sim > 0.5:  # Only consider meaningful similarities
                            similarities.append(sim)
                            relation_types.append(self._infer_relation_type(mem1, mem2, sim, model_name))
                    except Exception as e:
                        # Skip if embedding comparison fails
                        print(f"[Warning] Similarity calculation error: {e}")
                        continue

                # If we have valid similarities, record the connection
                if similarities:
                    avg_similarity = sum(similarities) / len(similarities)

                    # Only store significant connections (threshold could be adaptive)
                    if avg_similarity > 0.6:
                        # Find the most likely relation type based on frequency
                        if relation_types:
                            relation_type = max(set(relation_types), key=relation_types.count)
                        else:
                            relation_type = "semantic"

                        # Add bidirectional connections
                        self.connection_graph[mem1.id].append((mem2.id, avg_similarity, relation_type))
                        self.connection_graph[mem2.id].append((mem1.id, avg_similarity, relation_type))

        # Find memory clusters using community detection
        self._detect_memory_clusters()

        # Identify semantic fields across the memory space
        self._identify_semantic_fields(all_memories)

        print(
            f"[MemoryConnector] Found {len(self.connection_graph)} connected memories in {len(self.memory_clusters)} clusters")

    def _infer_relation_type(self, mem1, mem2, similarity, model_name):
        """
        Infer the likely relationship type between two memories.

        Types include:
        - temporal: likely time-related connection
        - causal: one memory may cause or follow from another
        - semantic: related by meaning
        - episodic: part of the same episode or story
        - conceptual: related abstract concepts
        """
        # Simple rule-based approach - could be enhanced with ML
        from embedding_utils import compute_adaptive_similarity
        # Temporal relation if creation times are close
        time_diff = abs((mem1.creation_time - mem2.creation_time).total_seconds())
        if time_diff < 300:  # 5 minutes
            return "temporal"

        # Check for same memory type
        if mem1.memory_type == mem2.memory_type:
            if mem1.memory_type == "Flashbulb":
                return "episodic"
            elif mem1.memory_type == "Procedural":
                return "causal"
            elif mem1.memory_type == "Liminal" or mem1.memory_type == "Generative":
                return "conceptual"

        # Default to semantic relationship
        return "semantic"

    def _detect_memory_clusters(self):
        """
        Use a simple clustering approach to find groups of related memories.

        This creates 'attractors' in the memory space that can be used
        to enhance retrieval by identifying coherent memory clusters.
        """
        visited = set()

        for memory_id in self.connection_graph:
            if memory_id in visited:
                continue

            # Start a new cluster
            cluster = []
            queue = [memory_id]

            while queue:
                current_id = queue.pop(0)
                if current_id in visited:
                    continue

                visited.add(current_id)
                cluster.append(current_id)

                # Add connected nodes to queue
                for connected_id, similarity, _ in self.connection_graph[current_id]:
                    if connected_id not in visited and similarity > 0.7:  # Higher threshold for cluster inclusion
                        queue.append(connected_id)

            if len(cluster) > 1:  # Only store non-trivial clusters
                self.memory_clusters.append(cluster)

    def _identify_semantic_fields(self, all_memories):
        """
        Identify semantic fields in the memory space.

        Semantic fields are regions in the embedding space where memories
        share similar conceptual meaning, creating a field-like structure
        in the memory landscape.
        """
        if not all_memories:
            return

        try:
            # Use clustering on embeddings to find semantic fields
            from sklearn.cluster import DBSCAN
            from embedding_utils import compute_adaptive_similarity

            # Instead of creating a numpy array directly, we'll process memories individually
            valid_memories = [mem for mem in all_memories if mem.embedding is not None]
            memory_ids = [mem.id for mem in valid_memories]

            if len(valid_memories) < 2:
                return

            # Create a distance matrix manually using our adaptive similarity function
            n_memories = len(valid_memories)
            distance_matrix = np.zeros((n_memories, n_memories))

            for i in range(n_memories):
                for j in range(i + 1, n_memories):
                    # Calculate similarity and convert to distance (1 - similarity)
                    similarity = compute_adaptive_similarity(
                        valid_memories[i].embedding,
                        valid_memories[j].embedding
                    )
                    distance = 1.0 - similarity

                    # Fill both sides of the symmetric matrix
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance

            # Use DBSCAN with precomputed distance matrix
            clustering = DBSCAN(
                eps=0.3,
                min_samples=2,
                metric='precomputed'
            ).fit(distance_matrix)

            # Create semantic fields from clusters
            labels = clustering.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            for i in range(n_clusters):
                cluster_indices = np.where(labels == i)[0]
                field_memories = [memory_ids[idx] for idx in cluster_indices]

                if field_memories:
                    # Find a representative name for this semantic field
                    field_name = f"semantic_field_{i}"
                    self.semantic_fields[field_name] = field_memories
        except Exception as e:
            print(f"[MemoryConnector] Error in semantic field identification: {e}")

    def enhance_retrieval(self, initial_memories, query, context=None):
        """
        Enhance memory retrieval using the connection graph, clusters, and semantic fields.

        [Rest of docstring unchanged]
        """
        # Import locally to avoid circular imports
        from embedding_utils import memory_similarity

        if not initial_memories:
            return []

        enhanced_memories = list(initial_memories)
        memory_ids = {mem.id for mem in enhanced_memories}

        # 1. Add directly connected memories with different connection types
        connected_memories = []
        connection_types_included = set()

        for memory in initial_memories:
            if memory.id in self.connection_graph:
                # Sort connections by similarity (highest first)
                connections = sorted(self.connection_graph[memory.id],
                                     key=lambda x: x[1], reverse=True)

                # Include diverse connection types for creativity
                for connected_id, similarity, relation_type in connections:
                    # Limit by connection type to ensure diversity
                    if connected_id not in memory_ids and relation_type not in connection_types_included:
                        connection_types_included.add(relation_type)

                        # Find the memory object
                        for mem_type in self.memory_blossom.memory_stores:
                            for mem in self.memory_blossom.memory_stores[mem_type]:
                                if mem.id == connected_id:
                                    connected_memories.append(mem)
                                    memory_ids.add(mem.id)
                                    break

        # 2. Add memories from identified clusters (emergent memory groups)
        cluster_memories = []
        for memory in initial_memories:
            for cluster in self.memory_clusters:
                if memory.id in cluster:
                    # Add a sample of other memories from this cluster
                    other_ids = [m_id for m_id in cluster if m_id != memory.id
                                 and m_id not in memory_ids][:2]

                    for other_id in other_ids:
                        if other_id not in memory_ids:
                            # Find the memory object
                            for mem_type in self.memory_blossom.memory_stores:
                                for mem in self.memory_blossom.memory_stores[mem_type]:
                                    if mem.id == other_id:
                                        cluster_memories.append(mem)
                                        memory_ids.add(mem.id)
                                        break

        # 3. Add memories from relevant semantic fields
        field_memories = []
        for field_name, field_ids in self.semantic_fields.items():
            # Check if this field is relevant to any initial memory
            if any(mem.id in field_ids for mem in initial_memories):
                # Add a sample from this field
                sample_ids = [m_id for m_id in field_ids if m_id not in memory_ids][:1]

                for sample_id in sample_ids:
                    # Find the memory object
                    for mem_type in self.memory_blossom.memory_stores:
                        for mem in self.memory_blossom.memory_stores[mem_type]:
                            if mem.id == sample_id:
                                field_memories.append(mem)
                                memory_ids.add(mem.id)
                                break

        # Combine all memory sources with initial memories first
        combined_memories = enhanced_memories

        # Add connected memories (direct associations)
        combined_memories.extend(connected_memories)

        # Add cluster memories (emergent groups)
        combined_memories.extend(cluster_memories)

        # Add field memories (conceptual regions)
        combined_memories.extend(field_memories)

        # Limit to a reasonable number (10 total)
        return combined_memories[:10]

    def get_memory_analysis(self, memories):
        """
        Analyze a set of memories to identify patterns and relationships.

        This provides meta-information about how memories are connected,
        which can be used to enhance the system's prompt and reasoning.

        Args:
            memories: List of memories to analyze

        Returns:
            Dictionary of analysis results
        """
        if not memories:
            return {"patterns": [], "connections": []}

        analysis = {
            "patterns": [],
            "connections": []
        }

        # Find memory type patterns
        mem_types = [mem.memory_type for mem in memories]
        type_counts = defaultdict(int)
        for t in mem_types:
            type_counts[t] += 1

        for t, count in type_counts.items():
            if count > 1:
                analysis["patterns"].append(f"Multiple {t} memories detected")

        # Find connections between memories
        for i, mem1 in enumerate(memories):
            for j, mem2 in enumerate(memories[i + 1:], i + 1):
                # Check if these memories are connected in our graph
                connected = False
                relation_type = None

                for conn_id, _, rel_type in self.connection_graph.get(mem1.id, []):
                    if conn_id == mem2.id:
                        connected = True
                        relation_type = rel_type
                        break

                if connected:
                    analysis["connections"].append({
                        "source": mem1.id,
                        "target": mem2.id,
                        "type": relation_type
                    })

        # Add cluster information
        for cluster in self.memory_clusters:
            # Check if multiple memories from this cluster are present
            cluster_memories = [mem for mem in memories if mem.id in cluster]
            if len(cluster_memories) > 1:
                analysis["patterns"].append(f"Memory group detected: {len(cluster_memories)} related memories")

        return analysis