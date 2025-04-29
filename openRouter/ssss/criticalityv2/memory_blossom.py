"""
MemoryBlossom: An Edge of Chaos AI Memory System

This module implements a sophisticated memory architecture designed to operate
in the "critical zone" between order and chaos - a sweet spot where responses
are both coherent and creative.

The system includes:
1. Multi-Modal Memory System - Different memory types with specialized embeddings
2. Memory Interconnections - A graph-like structure of related memories
3. Edge of Chaos Dynamics - Balancing order and creativity in memory retrieval
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from collections import defaultdict
import random

from dotenv import load_dotenv

load_dotenv()

# Import our memory connector (once it exists)
try:
    from memory_connector import MemoryConnector
except ImportError:
    print("[Warning] MemoryConnector module not found, will be initialized later")


class Memory:
    """
    A rich memory representation with multiple attributes and embeddings.

    Memories are more than just data - they have emotional significance,
    contextual relevance, and complex interconnections with other memories.
    """

    def __init__(self,
                 content: str,
                 memory_type: str,
                 metadata: Dict[str, Any] = None,
                 emotion_score: float = 0.0):
        """
        Create a memory with rich contextual information

        Args:
            content: The actual memory content
            memory_type: Type of memory (Explicit, Emotional, etc.)
            metadata: Additional context about the memory
            emotion_score: Emotional intensity of the memory
        """
        self.id = str(datetime.now().timestamp())
        self.content = content
        self.memory_type = memory_type
        self.metadata = metadata or {}

        # Emotional and contextual attributes
        self.emotion_score = emotion_score
        self.creation_time = datetime.now()
        self.last_accessed = datetime.now()
        self.access_count = .0
        self.decay_factor = 1.0  # Memory decay over time (1.0 = no decay)
        self.salience = 1.0  # Importance/prominence of the memory

        # Coherence and creativity measures (for criticality)
        self.coherence_score = 0.0
        self.novelty_score = 0.0

        # Embedding placeholders
        self.embedding = None  # Primary embedding
        self.contextual_embedding = None  # Context-aware embedding

        # Connections to other memories
        self.connections = []  # [(memory_id, strength, type)]

    def update_access(self):
        """Update memory access statistics and adjust salience."""
        self.last_accessed = datetime.now()
        self.access_count += 1

        # Increase salience with access (hebbian learning principle)
        self.salience = min(1.0, self.salience + 0.05)

    def decay(self, rate: float = 0.01):
        """Apply memory decay based on time elapsed since last access."""
        time_since_access = (datetime.now() - self.last_accessed).total_seconds()

        # Logarithmic decay function
        decay_amount = rate * np.log(1 + time_since_access / 86400)  # 86400 secs = 1 day
        self.decay_factor = max(0.1, self.decay_factor - decay_amount)

        # Adjust salience based on decay
        self.salience = self.salience * self.decay_factor

    def get_effective_salience(self) -> float:
        """Get the effective salience considering emotion, access count, and decay."""
        # Emotional memories decay more slowly
        emotion_factor = 1.0 + (self.emotion_score * 0.5)

        # Access count increases salience
        access_factor = min(2.0, 1.0 + (0.1 * self.access_count))

        return self.salience * emotion_factor * access_factor * self.decay_factor


class MemoryBlossom:
    """
    An advanced memory system operating at the edge of chaos.

    MemoryBlossom combines multiple specialized memory stores with different
    embedding models, implementing principles from complex systems theory to
    create a richer, more nuanced memory architecture.
    """

    def __init__(self, openai_client=None):
        """
        Initialize the MemoryBlossom system.

        Args:
            openai_client: Optional OpenAI client for LLM-based memory classification
        """
        self.openai_client = openai_client

        # Memory stores by type
        self.memory_stores = {
            "Explicit": [],  # Factual, declarative knowledge
            "Emotional": [],  # Memories with affective significance
            "Procedural": [],  # Knowledge about processes and methods
            "Flashbulb": [],  # Vivid, autobiographical memories
            "Somatic": [],  # Sensory or physical experiences
            "Liminal": [],  # Emergent, threshold concepts
            "Generative": []  # Creative or imaginative constructs
        }

        # Load specialized embedding models for each memory type
        try:
            self.embedding_models = {
                'Explicit': SentenceTransformer('BAAI/bge-small-en-v1.5'),
                'Emotional': SentenceTransformer('all-MiniLM-L6-v2'),
                'Procedural': SentenceTransformer('intfloat/e5-base-v2'),
                'Flashbulb': SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True),
                'Somatic': SentenceTransformer('all-MiniLM-L6-v2'),  # Ideally would be CLIP for multimodal
                'Liminal': SentenceTransformer('mixedbread-ai/mxbai-embed-xsmall-v1'),
                'Generative': SentenceTransformer('all-MiniLM-L6-v2'),
            }
        except Exception as e:
            print(f"[MemoryBlossom] Warning: Some embeddings failed to load: {e}")
            # Use fallback model for all memory types
            basic_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_models = {key: basic_model for key in [
                'Explicit', 'Emotional', 'Procedural', 'Flashbulb', 'Somatic', 'Liminal', 'Generative'
            ]}

        # Memory connector for discovering relationships between memories
        self.memory_connector = None

        # Criticality parameters
        self.memory_temperature = 0.7  # Controls randomness in memory selection
        self.coherence_bias = 0.6  # Bias toward coherent memory patterns
        self.novelty_bias = 0.4  # Bias toward novel/surprising memories

        # Meta-memory attributes
        self.memory_statistics = defaultdict(int)  # Track memory distributions
        self.memory_patterns = []  # Recognized patterns in memory access
        self.memory_transitions = defaultdict(int)  # Track memory type transitions

    def initialize_memory_connections(self):
        """Initialize the memory connector and analyze connections."""
        try:
            # Import here in case it wasn't available at module import time
            from memory_connector import MemoryConnector
            self.memory_connector = MemoryConnector(self)
            self.memory_connector.analyze_all_memories()
            print("[MemoryBlossom] Memory connections initialized and analyzed")
        except Exception as e:
            print(f"[MemoryBlossom] Error initializing memory connections: {e}")

    def dynamic_classify_memory(self, content: str) -> str:
        """
        Classify memory type using an LLM for nuanced understanding.

        This allows for more sophisticated memory categorization based on
        semantic understanding rather than simple keyword matching.

        Args:
            content: Text content to classify

        Returns:
            Memory type classification
        """
        if self.openai_client is None:
            print("[MemoryBlossom] Warning: No OpenAI client provided, using fallback classification.")
            return self.fallback_classify_memory(content)

        try:
            prompt = f"""Classify the following text into one of these categories:
            [Explicit, Emotional, Procedural, Flashbulb, Somatic, Liminal, Generative]

            Definitions:
            - Explicit: Factual, declarative knowledge
            - Emotional: Memories with affective significance
            - Procedural: Knowledge about processes and methods
            - Flashbulb: Vivid, autobiographical memories
            - Somatic: Sensory or physical experiences 
            - Liminal: Emergent, threshold concepts
            - Generative: Creative or imaginative constructs

            Text: {content}

            Only respond with a single word: the category name."""

            response = self.openai_client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            category = response.choices[0].message.content.strip()

            valid_categories = [
                "Explicit", "Emotional", "Procedural",
                "Flashbulb", "Somatic", "Liminal", "Generative"
            ]

            if category not in valid_categories:
                print(f"[MemoryBlossom] Warning: LLM returned unexpected category '{category}', using fallback.")
                return self.fallback_classify_memory(content)

            return category

        except Exception as e:
            print(f"[MemoryBlossom] Error in dynamic classification: {e}")
            return self.fallback_classify_memory(content)

    def fallback_classify_memory(self, content: str) -> str:
        """
        Fallback heuristic classification if no LLM is available.

        Uses keyword matching and linguistic patterns to estimate memory type.

        Args:
            content: Text content to classify

        Returns:
            Memory type classification
        """
        content_lower = content.lower()

        # Define classifiers with weighted patterns
        classifiers = {
            'Emotional': [
                (["feel", "emotion", "love", "fear", "happy", "sad", "joy", "angry"], 1.0),
                (["heart", "passionate", "excited", "moved", "touched"], 0.8),
                (["hate", "adore", "cry", "laugh", "smile", "tears"], 0.7)
            ],
            'Procedural': [
                (["how to", "steps", "method", "procedure", "process"], 1.0),
                (["first", "then", "next", "finally", "lastly", "step"], 0.8),
                (["instructions", "guide", "tutorial", "approach"], 0.7)
            ],
            'Flashbulb': [
                (["i remember", "my first", "identity", "life changing", "moment"], 1.0),
                (["birthday", "wedding", "funeral", "childhood", "memory"], 0.8),
                (["that day", "never forget", "vivid", "remember when"], 0.7)
            ],
            'Somatic': [
                (["taste", "smell", "sound", "color", "vision", "temperature"], 1.0),
                (["touch", "pain", "sensation", "feeling", "physical"], 0.8),
                (["hot", "cold", "loud", "soft", "bright", "dark"], 0.7)
            ],
            'Liminal': [
                (["maybe", "perhaps", "what if", "possibility", "emerging"], 1.0),
                (["between", "threshold", "transition", "edge", "boundary"], 0.8),
                (["becoming", "evolving", "shifting", "transforming"], 0.7)
            ],
            'Generative': [
                (["dream", "imagine", "fantasy", "poem", "create", "invent"], 1.0),
                (["story", "fiction", "narrative", "tale", "creativity"], 0.8),
                (["novel", "artistic", "visionary", "innovative"], 0.7)
            ]
        }

        # Initialize scores
        scores = defaultdict(float)

        # Calculate scores for each memory type
        for memory_type, patterns in classifiers.items():
            for keywords, weight in patterns:
                if any(keyword in content_lower for keyword in keywords):
                    scores[memory_type] += weight

        # If any scores, return highest scoring type
        if scores:
            highest_type = max(scores.items(), key=lambda x: x[1])[0]
            return highest_type

        # Default to Explicit if no patterns match
        return 'Explicit'

    def analyze_memory_patterns(self, query: str, top_k: int = 10):
        """
        Analyze patterns in memories to detect emergent properties and criticality.

        This helps the system become aware of its own memory structure and biases.

        Args:
            query: Current query to provide context
            top_k: Maximum number of patterns to return

        Returns:
            Dictionary of pattern analysis
        """
        # Get all memories
        memories = []
        for memory_type in self.memory_stores:
            memories.extend(self.memory_stores[memory_type])

        if not memories:
            return {"status": "no_memories_available"}

        # Calculate distribution of memory types
        memory_type_distribution = {
            mem_type: len(self.memory_stores[mem_type])
            for mem_type in self.memory_stores
        }

        # Calculate average emotion scores by type
        emotion_by_type = {}
        for mem_type in self.memory_stores:
            if self.memory_stores[mem_type]:
                emotion_by_type[mem_type] = sum(
                    mem.emotion_score for mem in self.memory_stores[mem_type]
                ) / len(self.memory_stores[mem_type])
            else:
                emotion_by_type[mem_type] = 0

        # Find connections between memory types using memory connector if available
        connections = []
        if self.memory_connector and hasattr(self.memory_connector, 'connection_graph'):
            for memory_id, connections_list in self.memory_connector.connection_graph.items():
                for target_id, similarity, relation_type in connections_list[:5]:  # Limit to top 5
                    # Find the memory types
                    source_type = None
                    target_type = None

                    for mem_type in self.memory_stores:
                        for mem in self.memory_stores[mem_type]:
                            if mem.id == memory_id:
                                source_type = mem_type
                            if mem.id == target_id:
                                target_type = mem_type

                            if source_type and target_type:
                                break

                    if source_type and target_type:
                        connections.append({
                            "source_type": source_type,
                            "target_type": target_type,
                            "similarity": float(similarity),
                            "relation_type": relation_type
                        })

        # Calculate memory age statistics
        now = datetime.now()
        age_stats = {
            "newest": min([now - mem.creation_time for mem in memories]).total_seconds(),
            "oldest": max([now - mem.creation_time for mem in memories]).total_seconds(),
            "average_age": sum([(now - mem.creation_time).total_seconds() for mem in memories]) / len(memories)
        }

        # Analyze memory criticality
        criticality_analysis = self._analyze_memory_criticality(memories)

        # Return comprehensive analysis
        return {
            "memory_type_distribution": memory_type_distribution,
            "emotion_by_type": emotion_by_type,
            "memory_connections": connections[:top_k],
            "memory_age_stats": age_stats,
            "criticality_analysis": criticality_analysis,
            "total_memories": len(memories),
            "analysis_time": datetime.now().isoformat()
        }

    def _analyze_memory_criticality(self, memories):
        """
        Analyze how memories are distributed in the order-chaos spectrum.

        This helps identify if the memory system is operating in the critical zone
        between excessive order (too predictable) and excessive chaos (too random).

        Args:
            memories: List of memories to analyze

        Returns:
            Dictionary of criticality metrics
        """
        if not memories:
            return {"criticality_score": 0.5}  # Default neutral

        # Calculate coherence vs novelty distribution
        coherence_scores = []
        novelty_scores = []

        for memory in memories:
            # Use memory attributes if available, otherwise estimate
            coherence = getattr(memory, 'coherence_score', 0)
            novelty = getattr(memory, 'novelty_score', 0)

            # If not set, estimate based on memory type
            if coherence == 0:
                if memory.memory_type in ["Explicit", "Procedural"]:
                    coherence = 0.8
                elif memory.memory_type in ["Emotional", "Flashbulb"]:
                    coherence = 0.6
                elif memory.memory_type in ["Liminal", "Generative"]:
                    coherence = 0.4
                else:
                    coherence = 0.5

            if novelty == 0:
                if memory.memory_type in ["Generative", "Liminal"]:
                    novelty = 0.8
                elif memory.memory_type in ["Emotional", "Somatic"]:
                    novelty = 0.6
                elif memory.memory_type in ["Explicit", "Procedural"]:
                    novelty = 0.3
                else:
                    novelty = 0.5

            coherence_scores.append(coherence)
            novelty_scores.append(novelty)

        # Calculate criticality metrics
        avg_coherence = sum(coherence_scores) / len(coherence_scores)
        avg_novelty = sum(novelty_scores) / len(novelty_scores)

        # Calculate criticality as a balance between coherence and novelty
        # Optimal criticality is when both are moderately high
        criticality_score = (avg_coherence * 0.6 + avg_novelty * 0.4)

        # Calculate zone (0=ordered, 1=critical, 2=chaotic)
        if avg_coherence > 0.7 and avg_novelty < 0.3:
            zone = 0  # Ordered zone (high coherence, low novelty)
        elif avg_coherence < 0.5 and avg_novelty > 0.6:
            zone = 2  # Chaotic zone (low coherence, high novelty)
        else:
            zone = 1  # Critical zone (balanced)

        # Calculate variance as a measure of complexity
        coherence_variance = np.var(coherence_scores)
        novelty_variance = np.var(novelty_scores)

        return {
            "criticality_score": criticality_score,
            "avg_coherence": avg_coherence,
            "avg_novelty": avg_novelty,
            "coherence_variance": coherence_variance,
            "novelty_variance": novelty_variance,
            "zone": zone  # 0=ordered, 1=critical, 2=chaotic
        }

    def add_memory(self,
                   content: str,
                   memory_type: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   emotion_score: float = 0.0) -> Memory:
        """
        Add a new memory to the appropriate memory store.

        Args:
            content: Memory content
            memory_type: Explicit memory type (optional)
            metadata: Additional context
            emotion_score: Emotional intensity

        Returns:
            Created Memory object
        """
        # Determine memory type if not provided
        if not memory_type:
            memory_type = self.dynamic_classify_memory(content)

        # Create memory object
        memory = Memory(
            content=content,
            memory_type=memory_type,
            metadata=metadata,
            emotion_score=emotion_score
        )

        # Embed the memory using appropriate model
        embedding_model = self.embedding_models[memory_type]
        memory.embedding = embedding_model.encode([content])[0]

        # Estimate coherence and novelty
        memory.coherence_score = self._estimate_coherence(content, memory_type)
        memory.novelty_score = self._estimate_novelty(content, memory.embedding)

        # Store in appropriate memory type collection
        self.memory_stores[memory_type].append(memory)

        # Update memory statistics
        self.memory_statistics[memory_type] += 1

        # Update memory connections if connector exists
        if self.memory_connector is not None:
            # Schedule a connection analysis after some threshold of new memories
            if sum(self.memory_statistics.values()) % 10 == 0:  # Every 10 memories
                self.memory_connector.analyze_all_memories()

        return memory

    def _estimate_coherence(self, content: str, memory_type: str) -> float:
        """
        Estimate the coherence of a memory based on its content and type.

        Higher coherence = more structured, logical, consistent

        Args:
            content: Memory content
            memory_type: Memory type

        Returns:
            Coherence score (0-1)
        """
        # Simple heuristics for coherence estimation
        word_count = len(content.split())
        sentence_count = len([s for s in content.split('.') if s.strip()])

        if sentence_count == 0:
            sentence_count = 1  # Avoid division by zero

        avg_sentence_length = word_count / sentence_count

        # Longer average sentences tend to be more structured/coherent
        length_factor = min(1.0, avg_sentence_length / 20)

        # Memory type coherence bias
        type_coherence = {
            "Explicit": 0.8,
            "Procedural": 0.9,
            "Flashbulb": 0.7,
            "Emotional": 0.6,
            "Somatic": 0.5,
            "Liminal": 0.4,
            "Generative": 0.3
        }

        # Connector words indicate logical structure
        connector_words = [
            "because", "therefore", "thus", "consequently", "however",
            "moreover", "furthermore", "since", "although", "despite"
        ]

        connector_factor = 0.0
        for word in connector_words:
            if word in content.lower():
                connector_factor += 0.1
        connector_factor = min(0.5, connector_factor)

        # Combine factors (weighted average)
        coherence = (
                length_factor * 0.3 +
                type_coherence.get(memory_type, 0.5) * 0.5 +
                connector_factor * 0.2
        )

        return coherence

    def _estimate_novelty(self, content: str, embedding: np.ndarray) -> float:
        """
        Estimate the novelty of a memory compared to existing memories.

        Higher novelty = more unique, surprising, creative

        Args:
            content: Memory content
            embedding: Memory embedding

        Returns:
            Novelty score (0-1)
        """
        # Import locally to avoid circular imports
        from embedding_utils import compute_adaptive_similarity

        # If no embedding or no existing memories, default to moderate novelty
        if embedding is None or sum(len(memories) for memories in self.memory_stores.values()) == 0:
            return 0.5

        # Calculate similarities to existing memories using adaptive approach
        similarities = []
        for memory_type, memories in self.memory_stores.items():
            for memory in memories:
                if memory.embedding is not None:
                    try:
                        # Calculate similarity using adaptive method
                        similarity = compute_adaptive_similarity(embedding, memory.embedding)
                        similarities.append(similarity)
                    except Exception as e:
                        print(f"[Warning] Similarity calculation error: {e}")
                        continue

        # If no similarities calculated, default to moderate novelty
        if not similarities:
            return 0.5

        # Higher max similarity = lower novelty
        max_similarity = max(similarities) if similarities else 0
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0

        # Calculate novelty as inverse of similarity (weighted)
        novelty = 1.0 - (max_similarity * 0.7 + avg_similarity * 0.3)

        return novelty

    def retrieve_memories(self,
                          query: str,
                          memory_type: Optional[str] = None,
                          top_k: int = 3) -> List[Memory]:
        """
        Retrieve most relevant memories using the edge of chaos principle.


        """
        # Import locally to avoid circular imports
        from embedding_utils import compute_adaptive_similarity

        # If no specific type, search all memory types
        search_types = [memory_type] if memory_type else list(self.memory_stores.keys())

        all_matches = []
        for mem_type in search_types:
            # Skip if no memories in this type
            if not self.memory_stores[mem_type]:
                continue

            # Use appropriate embedding model
            embedding_model = self.embedding_models[mem_type]

            # Embed query
            query_embedding = embedding_model.encode([query])[0]

            # Calculate similarities with appropriate model
            memories = self.memory_stores[mem_type]
            similarities = [
                compute_adaptive_similarity(query_embedding, memory.embedding)
                for memory in memories
            ]

            # Apply criticality to memory selection
            matches = self._apply_criticality_to_selection(memories, similarities)
            all_matches.extend(matches)

        # Sort by combined score and take top k
        all_matches.sort(key=lambda x: x[1], reverse=True)
        top_memories = [mem for mem, _ in all_matches[:top_k]]

        # Update access statistics for retrieved memories
        for memory in top_memories:
            memory.update_access()

        # Track memory transitions for statistical learning
        if len(top_memories) > 1:
            for i in range(len(top_memories) - 1):
                transition = (top_memories[i].memory_type, top_memories[i + 1].memory_type)
                self.memory_transitions[transition] += 1

        return top_memories

    def _apply_criticality_to_selection(self, memories, similarities) -> List[Tuple[Memory, float]]:
        """
        Apply edge of chaos principles to memory selection.

        This balances deterministic selection (highest similarity)
        with stochastic selection (controlled randomness) based on
        the system's temperature and biases.

        Args:
            memories: List of memories
            similarities: Corresponding similarity scores

        Returns:
            List of (memory, score) tuples
        """
        matches = []

        # For each memory, calculate a combined score
        for i, (memory, similarity) in enumerate(zip(memories, similarities)):
            # Base score is the similarity
            base_score = similarity

            # Apply coherence bias (favors memories with high coherence)
            coherence_component = memory.coherence_score * self.coherence_bias

            # Apply novelty bias (favors memories with high novelty)
            novelty_component = memory.novelty_score * self.novelty_bias

            # Apply salience adjustment (memories with higher effective salience)
            salience = memory.get_effective_salience()

            # Apply controlled randomness based on temperature
            noise = np.random.normal(0, self.memory_temperature)

            # Combine all components
            combined_score = (
                    base_score * 0.6 +  # Similarity is the primary component
                    coherence_component * 0.15 +
                    novelty_component * 0.15 +
                    salience * 0.1 +
                    noise * 0.05  # Small random component
            )

            matches.append((memory, combined_score))

        return matches

    def context_aware_retrieval(self,
                                query: str,
                                conversation_context: List[Dict[str, str]] = None) -> List[Memory]:
        """
        Enhanced memory retrieval using conversation context and memory connections.

        This creates a more holistic retrieval that considers the conversation flow
        and the interconnections between memories - operating at the edge of chaos.

        Args:
            query: Current query
            conversation_context: Recent conversation history

        Returns:
            List of relevant memories
        """
        # Get initial memories based on relevance to query
        initial_memories = self.retrieve_memories(query, top_k=5)

        # If conversation context available, use it to guide memory retrieval
        context_enhanced_memories = initial_memories
        if conversation_context:
            # Extract key context indicators from recent messages
            context_types = set()
            context_queries = []

            for msg in conversation_context[-3:]:  # Last 3 messages
                content = msg.get('content', '')
                if content:
                    memory_type = self.dynamic_classify_memory(content)
                    context_types.add(memory_type)
                    context_queries.append(content)

            # Retrieve additional memories based on conversation context
            if context_types and context_queries:
                # Create a combined context query
                context_query = " ".join(context_queries[-2:])  # Last 2 messages

                # Get context-specific memories
                context_memories = []
                for mem_type in context_types:
                    type_memories = self.retrieve_memories(context_query, mem_type, top_k=2)
                    context_memories.extend(type_memories)

                # Combine initial and context memories, removing duplicates
                seen_ids = {mem.id for mem in context_enhanced_memories}
                for memory in context_memories:
                    if memory.id not in seen_ids:
                        context_enhanced_memories.append(memory)
                        seen_ids.add(memory.id)

        # Use memory connector to enhance retrieval with connected memories
        if self.memory_connector is not None:
            # Get enhanced memories using the connector
            final_memories = self.memory_connector.enhance_retrieval(
                context_enhanced_memories, query, conversation_context
            )
            return final_memories

        return context_enhanced_memories

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics for meta-awareness."""
        stats = {
            "total_memories": sum(len(mems) for mems in self.memory_stores.values()),
            "memory_type_counts": {k: len(v) for k, v in self.memory_stores.items()},
            "access_pattern": dict(self.memory_transitions),
            "criticality": self._analyze_memory_criticality(
                [mem for mems in self.memory_stores.values() for mem in mems]
            )
        }
        return stats

    def decay_memories(self, rate: float = 0.01):
        """Apply decay to all memories based on time since last access."""
        for memory_type, memories in self.memory_stores.items():
            for memory in memories:
                memory.decay(rate)


# Memory persistence methods
def save_memories(memory_blossom: MemoryBlossom, filename: str = 'memory_blossom.json'):
    """Save memories to a JSON file"""
    try:
        # Get absolute path to ensure we know where it's saving
        full_path = os.path.abspath(filename)
        print(f"[MemoryBlossom] Attempting to save memories to: {full_path}")

        memories_data = {}
        for mem_type, memories in memory_blossom.memory_stores.items():
            memories_data[mem_type] = [
                {
                    'id': mem.id,
                    'content': mem.content,
                    'emotion_score': mem.emotion_score,
                    'coherence_score': mem.coherence_score,
                    'novelty_score': mem.novelty_score,
                    'metadata': mem.metadata,
                    'creation_time': mem.creation_time.isoformat(),
                    'last_accessed': mem.last_accessed.isoformat(),
                    'access_count': mem.access_count,
                    'salience': mem.salience,
                    'decay_factor': mem.decay_factor
                } for mem in memories
            ]

        # Add system metadata
        memories_data['_system'] = {
            'memory_statistics': dict(memory_blossom.memory_statistics),
            'memory_transitions': {str(k): v for k, v in memory_blossom.memory_transitions.items()},
            'criticality': {
                'temperature': memory_blossom.memory_temperature,
                'coherence_bias': memory_blossom.coherence_bias,
                'novelty_bias': memory_blossom.novelty_bias
            }
        }

        with open(filename, 'w') as f:
            json.dump(memories_data, f, indent=2)

        print(
            f"[MemoryBlossom] Saved {sum(len(mems) for mems in memory_blossom.memory_stores.values())} memories to {filename}")
        return True
    except Exception as e:
        print(f"[MemoryBlossom] ERROR saving memories: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def save_chat_history(conversation_history, filename='chat_history.json'):
    """Save chat history to a JSON file"""
    try:
        full_path = os.path.abspath(filename)
        print(f"[Chat] Attempting to save conversation history to: {full_path}")

        # Prepare data for serialization
        history_data = {
            'timestamp': datetime.now().isoformat(),
            'messages': conversation_history
        }

        with open(filename, 'w') as f:
            json.dump(history_data, f, indent=2)

        print(f"[Chat] Saved {len(conversation_history)} messages to {filename}")
        return True
    except Exception as e:
        print(f"[Chat] ERROR saving chat history: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def load_memories(memory_blossom: MemoryBlossom, filename: str = 'memory_blossom.json'):
    """Load memories from a JSON file"""
    try:
        with open(filename, 'r') as f:
            memories_data = json.load(f)

        # Load system metadata if available
        if '_system' in memories_data:
            system_data = memories_data.pop('_system')

            # Load statistics
            if 'memory_statistics' in system_data:
                memory_blossom.memory_statistics = defaultdict(int, system_data['memory_statistics'])

            # Load transitions
            if 'memory_transitions' in system_data:
                # Convert string tuple keys back to actual tuples
                transitions = {}
                for k_str, v in system_data['memory_transitions'].items():
                    # Parse string like "('Explicit', 'Emotional')"
                    k_str = k_str.strip("()").replace("'", "")
                    parts = [p.strip() for p in k_str.split(',')]
                    if len(parts) == 2:
                        transitions[(parts[0], parts[1])] = v
                memory_blossom.memory_transitions = defaultdict(int, transitions)

            # Load criticality settings
            if 'criticality' in system_data:
                crit_data = system_data['criticality']
                memory_blossom.memory_temperature = crit_data.get('temperature', 0.7)
                memory_blossom.coherence_bias = crit_data.get('coherence_bias', 0.6)
                memory_blossom.novelty_bias = crit_data.get('novelty_bias', 0.4)

        # Load memories
        for mem_type, memories in memories_data.items():
            for mem_data in memories:
                memory = Memory(
                    content=mem_data['content'],
                    memory_type=mem_type,
                    metadata=mem_data.get('metadata', {}),
                    emotion_score=mem_data.get('emotion_score', 0.0)
                )

                # Restore memory attributes
                memory.id = mem_data['id']
                memory.creation_time = datetime.fromisoformat(mem_data['creation_time'])
                memory.last_accessed = datetime.fromisoformat(mem_data['last_accessed'])
                memory.access_count = mem_data.get('access_count', 0)
                memory.coherence_score = mem_data.get('coherence_score', 0.5)
                memory.novelty_score = mem_data.get('novelty_score', 0.5)
                memory.salience = mem_data.get('salience', 1.0)
                memory.decay_factor = mem_data.get('decay_factor', 1.0)

                # Re-embed the memory
                embedding_model = memory_blossom.embedding_models[mem_type]
                memory.embedding = embedding_model.encode([memory.content])[0]

                memory_blossom.memory_stores[mem_type].append(memory)

        print(
            f"[MemoryBlossom] Loaded {sum(len(mems) for mems in memory_blossom.memory_stores.values())} memories from {filename}")

        # Initialize memory connections after loading memories
        memory_blossom.initialize_memory_connections()

    except FileNotFoundError:
        print(f"[MemoryBlossom] No saved memories found at {filename}")