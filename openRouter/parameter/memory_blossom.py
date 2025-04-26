import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai


class Memory:
    def __init__(self,
                 content: str,
                 memory_type: str,
                 metadata: Dict[str, Any] = None,
                 emotion_score: float = 0.0):
        """
        Create a memory with rich contextual information

        :param content: The actual memory content
        :param memory_type: Type of memory (Explicit, Emotional, etc.)
        :param metadata: Additional context about the memory
        :param emotion_score: Emotional intensity of the memory
        """
        self.id = str(datetime.now().timestamp())
        self.content = content
        self.memory_type = memory_type
        self.metadata = metadata or {}

        # Emotional and contextual attributes
        self.emotion_score = emotion_score
        self.creation_time = datetime.now()
        self.last_accessed = datetime.now()
        self.access_count = 0

        # Embedding placeholders
        self.embedding = None
        self.contextual_embedding = None


class MemoryBlossom:
    def __init__(self, openai_client=None):
        self.openai_client = openai_client

        self.memory_stores = {
            "Explicit": [],
            "Emotional": [],
            "Procedural": [],
            "Flashbulb": [],
            "Somatic": [],
            "Liminal": [],
            "Generative": []
        }


        try:
            self.embedding_models = {
                'Explicit': SentenceTransformer('BAAI/bge-small-en-v1.5'),
                'Emotional': SentenceTransformer('all-MiniLM-L6-v2'),
                'Procedural': SentenceTransformer('intfloat/e5-base-v2'),
                'Flashbulb': SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True),
                'Somatic': SentenceTransformer('all-MiniLM-L6-v2'), # THIS SHOULD BE CLIP EMBEDDING MODEL BUT ITS TOO HEAVY
                'Liminal': SentenceTransformer('mixedbread-ai/mxbai-embed-xsmall-v1'),
                'Generative': SentenceTransformer('all-MiniLM-L6-v2'),
            }
        except Exception as e:
            print(f"[MemoryBlossom] Warning: Some embeddings failed to load: {e}")
            # fallback simples para evitar crash
            basic_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_models = {key: basic_model for key in [
                'Explicit', 'Emotional', 'Procedural', 'Flashbulb', 'Somatic', 'Liminal', 'Generative'
            ]}

    def dynamic_classify_memory(self, content: str) -> str:
        """
        Uses a small LLM to classify the memory type dynamically.
        """

        if self.openai_client is None:
            print("[MemoryBlossom] Warning: No OpenAI client provided, using fallback classification.")
            return self.fallback_classify_memory(content)

        try:
            prompt = f"""Classify the following text into one of these categories:
            [Explicit, Emotional, Procedural, Flashbulb, Somatic, Liminal, Generative]

            Text: {content}

            Only respond with a single word: the category name."""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
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
        Fallback heuristic if no LLM is available or error occurs.
        """
        if any(keyword in content.lower() for keyword in ["feel", "emotion", "love", "fear", "happy", "sad"]):
            return 'Emotional'
        if any(keyword in content.lower() for keyword in ["how to", "steps", "method", "procedure"]):
            return 'Procedural'
        if any(keyword in content.lower() for keyword in ["i remember", "my first", "identity", "life changing"]):
            return 'Flashbulb'
        if any(keyword in content.lower() for keyword in ["taste", "smell", "sound", "color", "vision", "temperature"]):
            return 'Somatic'
        if any(keyword in content.lower() for keyword in ["maybe", "perhaps", "what if", "possibility", "emerging"]):
            return 'Liminal'
        if any(keyword in content.lower() for keyword in ["dream", "imagine", "fantasy", "poem", "create"]):
            return 'Generative'
        return 'Explicit'  # fallback

    def add_memory(self,
                   content: str,
                   memory_type: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   emotion_score: float = 0.0) -> Memory:
        """
        Add a new memory to the appropriate memory store

        :param content: Memory content
        :param memory_type: Explicit memory type (optional)
        :param metadata: Additional context
        :param emotion_score: Emotional intensity
        :return: Created Memory object
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

        # Embed the memory
        embedding_model = self.embedding_models[memory_type]
        memory.embedding = embedding_model.encode([content])[0]

        # Store in appropriate memory type collection
        self.memory_stores[memory_type].append(memory)

        return memory

    def retrieve_memories(self,
                          query: str,
                          memory_type: Optional[str] = None,
                          top_k: int = 3) -> List[Memory]:
        """
        Retrieve most relevant memories

        :param query: Search query
        :param memory_type: Specific memory type to search (optional)
        :param top_k: Number of memories to retrieve
        :return: List of most relevant memories
        """
        # If no specific type, search all memory types
        search_types = [memory_type] if memory_type else list(self.memory_stores.keys())

        all_memories = []
        for mem_type in search_types:
            # Skip if no memories in this type
            if not self.memory_stores[mem_type]:
                continue

            # Use appropriate embedding model
            embedding_model = self.embedding_models[mem_type]

            # Embed query
            query_embedding = embedding_model.encode([query])[0]

            # Calculate similarities
            memories = self.memory_stores[mem_type]
            similarities = [
                cosine_similarity([query_embedding], [memory.embedding])[0][0]
                for memory in memories
            ]

            # Sort memories by similarity
            sorted_memories = [
                mem for _, mem in sorted(
                    zip(similarities, memories),
                    key=lambda x: x[0],
                    reverse=True
                )
            ]

            # Add to overall memories list
            all_memories.extend(sorted_memories[:top_k])

        # Sort all memories and return top k
        all_memories.sort(key=lambda x: x.emotion_score, reverse=True)
        return all_memories[:top_k]

    def context_aware_retrieval(self,
                                query: str,
                                conversation_context: List[Dict[str, str]] = None) -> List[Memory]:
        """
        Retrieve memories using context-aware strategy

        :param query: Current query
        :param conversation_context: Recent conversation history
        :return: Relevant memories
        """
        # Analyze conversation context to determine memory types
        if conversation_context:
            # Extract key context indicators
            context_types = set()
            for msg in conversation_context[-3:]:  # Last 3 messages
                memory_type = self.dynamic_classify_memory(msg.get('content', ''))
                context_types.add(memory_type)

            # Prioritize memory types based on context
            if context_types:
                memories = []
                for mem_type in context_types:
                    type_memories = self.retrieve_memories(query, mem_type, top_k=2)
                    memories.extend(type_memories)

                return memories

        # Fallback to standard retrieval
        return self.retrieve_memories(query)


# Optional: Memory persistence methods
def save_memories(memory_blossom: MemoryBlossom, filename: str = 'memory_blossom.json'):
    """Save memories to a JSON file"""
    memories_data = {}
    for mem_type, memories in memory_blossom.memory_stores.items():
        memories_data[mem_type] = [
            {
                'id': mem.id,
                'content': mem.content,
                'emotion_score': mem.emotion_score,
                'metadata': mem.metadata,
                'creation_time': mem.creation_time.isoformat(),
                'last_accessed': mem.last_accessed.isoformat()
            } for mem in memories
        ]

    with open(filename, 'w') as f:
        json.dump(memories_data, f, indent=2)



def load_memories(memory_blossom: MemoryBlossom, filename: str = 'memory_blossom.json'):
    """Load memories from a JSON file"""
    try:
        with open(filename, 'r') as f:
            memories_data = json.load(f)

        for mem_type, memories in memories_data.items():
            for mem_data in memories:
                memory = Memory(
                    content=mem_data['content'],
                    memory_type=mem_type,
                    metadata=mem_data.get('metadata', {}),
                    emotion_score=mem_data.get('emotion_score', 0.0)
                )
                memory.id = mem_data['id']
                memory.creation_time = datetime.fromisoformat(mem_data['creation_time'])
                memory.last_accessed = datetime.fromisoformat(mem_data['last_accessed'])

                # Re-embed the memory
                embedding_model = memory_blossom.embedding_models[mem_type]
                memory.embedding = embedding_model.encode([memory.content])[0]

                memory_blossom.memory_stores[mem_type].append(memory)
    except FileNotFoundError:
        print("No saved memories found.")