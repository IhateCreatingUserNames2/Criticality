import os
import numpy as np
from typing import List, Dict, Optional
from dotenv import load_dotenv
from memory_blossom import MemoryBlossom, save_memories, load_memories
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
from openai import OpenAI

class MemoryBlossomChatbot:
    def __init__(
        self,
        openrouter_api_key: Optional[str] = None,
        model: str = "microsoft/mai-ds-r1:free",  # Now using OpenRouter model format
        memory_persistence: bool = True,
    ):
        """
        Initialize the chatbot with the MemoryBlossom system.

        :param openrouter_api_key: OpenRouter API key (uses env var if not provided)
        :param model: OpenRouter model to use (format: "provider/model-name")
        :param memory_persistence: Whether to save/load memories
        """
        # Set up OpenAI client with OpenRouter base URL
        self.openai_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        )

        # Rest of initialization remains the same
        self.memory_blossom = MemoryBlossom(openai_client=self.openai_client)

        if memory_persistence:
            load_memories(self.memory_blossom)

        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history_length = 10
        self.temperature = 0.7

        self._initialize_personality_memories()

    # ------------------------------------------------------------------ #
    #                        Helper / utility methods                    #
    # ------------------------------------------------------------------ #
    def _initialize_personality_memories(self):
        """Add some initial personality-defining memories."""
        personality_memories = [
            {
                "content": (
                    "I am an AI assistant with a deep commitment to "
                    "understanding and helping users."
                ),
                "type": "Flashbulb",
                "emotion_score": 0.8,
            },
            {
                "content": (
                    "My goal is to provide meaningful, contextually aware "
                    "responses that go beyond simple information retrieval."
                ),
                "type": "Emotional",
                "emotion_score": 0.6,
            },
        ]

        for mem in personality_memories:
            self.memory_blossom.add_memory(
                content=mem["content"],
                memory_type=mem["type"],
                emotion_score=mem["emotion_score"],
            )

    def _prepare_system_prompt(self, context_memories: List) -> str:
        """Build the system prompt that injects retrieved memories."""
        system_prompt = (
            "You are an advanced AI assistant with a multi-layered memory "
            "system.\n\n"
        )

        if context_memories:
            system_prompt += "Relevant Contextual Memories:\n"
            for mem in context_memories:
                system_prompt += f"- {mem.content} (Type: {mem.memory_type})\n"
            system_prompt += (
                "\nUse these memories to inform your response while maintaining "
                "natural conversation flow.\n\n"
            )

        return system_prompt

    def _calculate_emotion_score(self, text: str) -> float:
        """
        Very simple keyword-based emotion scoring.
        Replace with a real classifier if you need accuracy.
        """
        emotion_keywords = {
            "positive": [
                "happy",
                "joy",
                "love",
                "excited",
                "wonderful",
                "great",
                "amazing",
            ],
            "negative": [
                "sad",
                "angry",
                "frustrated",
                "disappointed",
                "upset",
                "worried",
            ],
            "intense": [
                "passionate",
                "deeply",
                "incredibly",
                "extremely",
                "profound",
            ],
        }

        text_lower = text.lower()
        score = 0.0
        if any(w in text_lower for w in emotion_keywords["positive"]):
            score += 0.3
        if any(w in text_lower for w in emotion_keywords["negative"]):
            score -= 0.2
        if any(w in text_lower for w in emotion_keywords["intense"]):
            score += 0.2
        return max(min(score, 1.0), -1.0)

    def retrieve_relevant_memories(
        self, query: str, top_k: int = 3
    ) -> List[Dict]:
        """Public helper for debugging -- returns plain-dict memories."""
        memories = self.memory_blossom.retrieve_memories(query, top_k=top_k)
        return [
            {
                "content": mem.content,
                "type": mem.memory_type,
                "emotion_score": mem.emotion_score,
                "timestamp": mem.creation_time.isoformat(),
            }
            for mem in memories
        ]

    def clear_conversation_history(self):
        """Reset the running chat context (does *not* wipe memories)."""
        self.conversation_history = []

    def add_memory(
        self,
        content: str,
        memory_type: Optional[str] = None,
        emotion_score: float = 0.0,
    ):
        """Manually insert a memory, then persist to disk."""
        self.memory_blossom.add_memory(
            content=content,
            memory_type=memory_type,
            emotion_score=emotion_score,
        )
        save_memories(self.memory_blossom)

    # ------------------------------------------------------------------ #
    #                               Chat                                 #
    # ------------------------------------------------------------------ #
    def chat(self, user_message: str) -> str:
        """Main entry point: send `user_message`, get assistant reply."""
        # 1) Pull relevant memories
        context_memories = self.memory_blossom.context_aware_retrieval(
            user_message, self.conversation_history
        )

        # 2) Build message list
        messages = [
            {
                "role": "system",
                "content": self._prepare_system_prompt(context_memories),
            }
        ]
        messages.extend(
            {
                "role": msg["role"],
                "content": msg["content"],
            }
            for msg in self.conversation_history[-self.max_history_length:]
        )
        messages.append({"role": "user", "content": user_message})

        # 3) Call OpenRouter (using OpenAI client with modified base URL)
        try:
            # Add OpenRouter-specific extra headers
            extra_headers = {
                "HTTP-Referer": os.getenv("YOUR_SITE_URL", "https://yourapp.com"),  # Optional
                "X-Title": os.getenv("YOUR_SITE_NAME", "MemoryBlossomChatbot"),  # Optional
            }

            # First generation
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                logprobs=True,
                top_logprobs=5,
                extra_headers=extra_headers
            )
            ai_response = response.choices[0].message.content
            logprobs_data = response.choices[0].logprobs.content
            criticality_zone = assess_criticality(ai_response, user_message, logprobs_data)

            # Regeneration loop remains the same
            max_regeneration_attempts = 3
            attempt = 0

            while criticality_zone != 1 and attempt < max_regeneration_attempts:
                attempt += 1
                print(f"[Regeneration Attempt {attempt}] Adjusting and retrying...")

                if criticality_zone == 0:
                    self.temperature = min(1.2, self.temperature + 0.2)
                elif criticality_zone == 2:
                    self.temperature = max(0.2, self.temperature - 0.2)

                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    logprobs=True,
                    top_logprops=5,
                    extra_headers=extra_headers
                )
                ai_response = response.choices[0].message.content
                logprobs_data = response.choices[0].logprobs.content
                criticality_zone = assess_criticality(ai_response, user_message, logprobs_data)

        except Exception as e:
            print(f"Error generating chat: {e}")
            return (
                "I'm having trouble responding right now -- please try again "
                "in a moment."
            )

        # Rest of method remains the same
        criticality_zone = assess_criticality(ai_response, user_message, logprobs_data)

        if criticality_zone == 0:
            print("[Criticality] Ordered Zone: Coherent but Predictable.")
        elif criticality_zone == 1:
            print("[Criticality] Critical Zone: Coherent and Novel! (Sweet Spot)")
        elif criticality_zone == 2:
            print("[Criticality] Chaotic Zone: Novel but Incoherent.")

        self.memory_blossom.add_memory(
            content=ai_response,
            memory_type=self.memory_blossom.dynamic_classify_memory(ai_response),
            emotion_score=self._calculate_emotion_score(ai_response),
        )

        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append(
            {"role": "assistant", "content": ai_response}
        )

        save_memories(self.memory_blossom)

        return ai_response


# ---------------------------------------------------------------------- #
#                        Simple command-line demo                        #
# ---------------------------------------------------------------------- #
def interactive_chat_cli():
    """Tiny REPL so you can test in a terminal."""
    bot = MemoryBlossomChatbot()
    print(
        "MemoryBlossom Chatbot\n"
        "  › type 'memories' to view recent memories\n"
        "  › type 'clear'    to wipe conversation context\n"
        "  › type 'exit'     to quit"
    )

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() == "exit":
                print("Goodbye!")
                break
            if user_input.lower() == "memories":
                mems = bot.retrieve_relevant_memories("recent", top_k=5)
                print("\n--- Recent Memories ---")
                for i, mem in enumerate(mems, 1):
                    print(f"{i}. [{mem['type']}] {mem['content']}")
                continue
            if user_input.lower() == "clear":
                bot.clear_conversation_history()
                print("Conversation history cleared.")
                continue

            print("\nAI:", bot.chat(user_input))
        except KeyboardInterrupt:
            print("\nInterrupted. Type 'exit' to quit.")



# --- Criticality Utilities ---

def measure_statistical_likelihood_from_logprobs(logprobs_data) -> float:
    """
    Measures how predictable (ordered) the model output is, based on real logprobs.
    Higher average logprobs = more confident = more ordered.
    """
    if not logprobs_data:
        return 0.5  # fallback to neutral if no data

    logprob_values = [token.logprob for token in logprobs_data if token.logprob is not None]
    if not logprob_values:
        return 0.5

    avg_logprob = np.mean(logprob_values)

    # Normalize: from (-inf, 0) to [0, 1]
    # Example: -5 very uncertain (~0.0), 0 is 100% confident (~1.0)
    normalized = np.clip((avg_logprob + 5) / 5, 0, 1)
    return normalized


def measure_semantic_novelty(text: str, query: str, strategy: str = "worst") -> float:
    """
    Measure novelty using available embedding models.
    strategy: "avg" (average similarity) or "worst" (minimum similarity)
    """

    memory_blossom = MemoryBlossom()

    explicit_model = memory_blossom.embedding_models['Explicit']
    emotional_model = memory_blossom.embedding_models['Emotional']

    text_embedding_explicit = explicit_model.encode([text])[0]
    query_embedding_explicit = explicit_model.encode([query])[0]

    text_embedding_emotional = emotional_model.encode([text])[0]
    query_embedding_emotional = emotional_model.encode([query])[0]

    similarity_explicit = cosine_similarity([text_embedding_explicit], [query_embedding_explicit])[0][0]
    similarity_emotional = cosine_similarity([text_embedding_emotional], [query_embedding_emotional])[0][0]

    if strategy == "avg":
        similarity = (similarity_explicit + similarity_emotional) / 2
    elif strategy == "worst":
        similarity = min(similarity_explicit, similarity_emotional)
    else:
        raise ValueError("Invalid strategy: choose 'avg' or 'worst'.")

    novelty = 1.0 - similarity

    return novelty


def measure_internal_consistency(text: str) -> float:
    """
    Rough internal coherence check based on sentence structure.
    Later you can replace this with LLM-based scoring.
    """
    sentences = text.split(".")
    if not sentences:
        return 0.0
    coherent_sentences = [s for s in sentences if len(s.strip()) > 20]
    return len(coherent_sentences) / max(len(sentences), 1)

def assess_criticality(model_output: str, query: str, logprobs_data) -> int:
    """
    Assesses where on the order-chaos spectrum the model output lies.
    0 = Ordered (coherent but predictable)
    1 = Critical (coherent and novel) <-- sweet spot
    2 = Chaotic (novel but incoherent)
    """

    pretrain_score = measure_statistical_likelihood_from_logprobs(logprobs_data)
    novelty_score = measure_semantic_novelty(model_output, query, strategy="worst")

    coherence_score = measure_internal_consistency(model_output)

    print(f"[Debug] Pretrain Score (Likelihood): {pretrain_score:.3f}")
    print(f"[Debug] Novelty Score (Semantic): {novelty_score:.3f}")
    print(f"[Debug] Coherence Score (Internal): {coherence_score:.3f}")

    NOVELTY_CRITICAL_THRESHOLD = 0.18

    if coherence_score > 0.7 and novelty_score > NOVELTY_CRITICAL_THRESHOLD:
        return 1  # Critical zone (sweet spot)
    elif coherence_score > 0.7 and novelty_score <= NOVELTY_CRITICAL_THRESHOLD:
        return 0  # Ordered zone (predictable)
    else:
        return 2  # Chaotic zone (messy)






if __name__ == "__main__":
    interactive_chat_cli()
