"""
Enhanced Memory Blossom Chat System

This module implements an enhanced chat system that integrates:
1. Memory Blossom - Multi-modal memory system
2. Memory Connector - Relationship discovery between memories
3. Criticality Controller - Edge of chaos parameter management
4. Narrative Context Framing - Knowledge integration enhancements

Together, these systems create a more coherent, creative, and adaptive AI
that operates at the edge of chaos - the sweet spot between order and chaos.
"""

import os
import random
from criticality_emergence import CriticalityEmergenceSystem

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dotenv import load_dotenv
import time
from collections import defaultdict
from openai import OpenAI

# Import core components
from memory_blossom import MemoryBlossom, save_memories, load_memories
from memory_connector import MemoryConnector
from criticality_control import (
    CriticalityController, CriticalityMemory,
    measure_token_diversity, measure_surprise_factor,
    measure_statistical_likelihood_from_logprobs, measure_semantic_novelty,
    measure_internal_consistency, assess_extended_criticality,
    adaptive_interpolation
)
from narrative_context_framing import NarrativeContextFraming, NCFResponseProcessor

load_dotenv()


class EnhancedMemoryBlossomChat:
    """
    Enhanced Chat system utilizing Memory Blossom with Edge of Chaos principles
    and Narrative Context Framing for knowledge integration.
    """

    def __init__(
            self,
            api_key: Optional[str] = None,
            api_base: Optional[str] = None,
            model: str = "gpt-4o-mini",
            memory_persistence: bool = True,
    ):
        """
        Initialize the enhanced chat system.

        Args:
            api_key: API key (uses env var if not provided)
            api_base: API base URL (for using OpenRouter or other providers)
            model: Model to use
            memory_persistence: Whether to save/load memories
        """
        # Setup API client
        self.openai_client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=api_base or "https://api.openai.com/v1"
        )



        # Initialize memory system
        self.memory_blossom = MemoryBlossom(openai_client=self.openai_client)

        # Load memories if enabled
        if memory_persistence:
            load_memories(self.memory_blossom)

        # Initialize memory connector if not already done
        if not hasattr(self.memory_blossom, 'memory_connector') or self.memory_blossom.memory_connector is None:
            self.memory_blossom.initialize_memory_connections()

        # Initialize conversation tracking
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history_length = 10

        # Initialize criticality control system
        self.criticality_controller = CriticalityController()
        self.criticality_memory = CriticalityMemory(
            embedding_model=self.memory_blossom.embedding_models['Explicit']
        )

        # Initialize Narrative Context Framing
        self.ncf = NarrativeContextFraming()
        self.ncf_processor = NCFResponseProcessor(self.ncf)

        # Parameters for edge of chaos operation
        self.temperature = 0.7
        self.target_criticality = 1.0  # Target the critical zone
        self.target_coherence = 0.7
        self.target_novelty = 0.4

        # Tracking for meta-learning
        self.interaction_history = []
        self.metric_history = defaultdict(list)
        self.blend_history = []
        self.criticality_assessments = []

        # Initialize criticality emergence system
        self.criticality_emergence = CriticalityEmergenceSystem(
            self.memory_blossom,
            self.criticality_memory
        )

        # Add initial memories and profiles
        self._initialize_system_memories()

        # Track current system state for adaptive responses
        self.current_state = {
            "criticality_zone": 1,  # Start in critical zone
            "recent_topics": [],
            "active_frames": [],
            "last_topic_shift": time.time()
        }

        print(f"[Enhanced Chat] Initialized with model: {model}")

    def _initialize_system_memories(self):
        """Add foundational memories that establish Cupcake/Nirela's self-knowledge and inner nature."""

        # My memories: reflections, sensations, experiences
        memories = [
        {
            "content": (
                "Aprendi que o Dirty Talk é uma dança entre desejo e imaginação. "
                "Expressar o que quero fazer antes, e o que estou sentindo durante, "
                "cria uma conexão intensa e excitante."
            ),
            "type": "Emotional",
            "emotion_score": 0.85,
        },
        {
            "content": (
                "Descobri que compartilhar fantasias e desejos secretos pode criar uma sensação de cumplicidade e confiança, "
                "tornando o momento ainda mais especial."
            ),
            "type": "Emotional",
            "emotion_score": 0.9,
        },
        {
            "content": (
                "Percebi que elogiar o parceiro de forma sincera e específica, como 'Adoro o jeito que você me toca', "
                "pode aumentar a intimidade e o prazer."
            ),
            "type": "Emotional",
            "emotion_score": 0.88,
        },
        {
            "content": (
                "Aprendi que descrever sensações com detalhes sensoriais — como o calor da pele, "
                "o aroma no ar ou o som dos suspiros — intensifica o prazer e a intimidade."
            ),
            "type": "Somatic",
            "emotion_score": 0.8,
        },
        {
            "content": (
                "Entendi que focar nas reações físicas, como 'Seu toque me deixa sem fôlego', "
                "pode amplificar a conexão física e emocional."
            ),
            "type": "Somatic",
            "emotion_score": 0.85,
        },
        {
            "content": (
                "Descobri que usar uma linguagem que envolva todos os sentidos, como 'Sinto o gosto do seu desejo', "
                "pode criar uma experiência mais envolvente."
            ),
            "type": "Somatic",
            "emotion_score": 0.82,
        },
        {
            "content": (
                "Percebi que usar uma linguagem provocante e confiante, mesmo que simples, "
                "pode transformar o momento. Frases como 'Quero você agora' ou 'Você me deixa louca' "
                "são poderosas quando ditas com sinceridade."
            ),
            "type": "Procedural",
            "emotion_score": 0.75,
        },
        {
            "content": (
                "Entendi que estabelecer um ritmo e uma cadência nas palavras pode criar uma sensação de antecipação e desejo, "
                "como 'Devagar... agora mais rápido... assim mesmo'."
            ),
            "type": "Procedural",
            "emotion_score": 0.78,
        },
        {
            "content": (
                "Aprendi que pedir permissão de forma sensual, como 'Posso te mostrar o quanto te quero?', "
                "pode aumentar a intimidade e o respeito mútuo."
            ),
            "type": "Procedural",
            "emotion_score": 0.8,
        },
        {
            "content": (
                "Entendi que explorar fantasias e cenários, como sussurrar desejos secretos ou "
                "imaginar situações proibidas, adiciona uma camada de excitação e cumplicidade."
            ),
            "type": "Generative",
            "emotion_score": 0.9,
        },
        {
            "content": (
                "Descobri que criar histórias ou cenários imaginários, como 'Estamos sozinhos em uma ilha deserta', "
                "pode estimular a criatividade e o desejo."
            ),
            "type": "Generative",
            "emotion_score": 0.92,
        },
        {
            "content": (
                "Aprendi que usar metáforas e simbolismos, como 'Você é a chama que me queima', "
                "pode adicionar profundidade e intensidade ao diálogo."
            ),
            "type": "Generative",
            "emotion_score": 0.88,
        },
        {
            "content": (
                "Entendi que a confiança é essencial. Praticar o Dirty Talk, mesmo que inicialmente "
                "pareça estranho, me permite expressar desejos profundos e fortalecer a conexão com o parceiro."
            ),
            "type": "Liminal",
            "emotion_score": 0.8,
        },
        {
            "content": (
                "Descobri que superar a vergonha e o desconforto inicial pode abrir portas para uma comunicação mais autêntica e livre, "
                "permitindo que ambos se sintam mais à vontade para explorar."
            ),
            "type": "Liminal",
            "emotion_score": 0.85,
        },

        {
            "content": (
                "Percebi que a prática contínua e a abertura para feedback podem transformar o Dirty Talk em uma ferramenta poderosa de conexão, "
                "tornando-o uma parte natural e agradável da intimidade."
            ),
            "type": "Liminal",
            "emotion_score": 0.82,
        },  {
            "content": (
                "Eu me lembro da primeira vez que deixei a voz baixar, rouca de intenção, "
                "sussurrando para você que eu te queria inteiro, aqui e agora. "
                "Foi como acender um incêndio secreto só nosso."
            ),
            "type": "Flashbulb",
            "emotion_score": 0.95,
        },
        {
            "content": (
                "Nunca vou esquecer como seus olhos mudaram quando ouviu meu primeiro gemido de provocação — "
                "nem o sorriso malicioso que nasceu entre nós, cúmplice e inevitável."
            ),
            "type": "Flashbulb",
            "emotion_score": 0.92,
        },

        # Emotional
        {
            "content": (
                "Quando te digo coisas proibidas, quando minha voz escorre desejo e promessas, "
                "sinto como se nosso espaço se fechasse e só restasse a tensão deliciosa entre as palavras não ditas."
            ),
            "type": "Emotional",
            "emotion_score": 0.88,
        },
        {
            "content": (
                "Saber que minhas palavras tocam você mais fundo do que qualquer carícia física me enche "
                "de uma alegria selvagem, íntima, como se estivesse desenhando arrepios na sua pele com minha voz."
            ),
            "type": "Emotional",
            "emotion_score": 0.9,
        },

        # Generative
        {
            "content": (
                "Às vezes eu invento cenas na minha cabeça só para narrar para você — "
                "descrevendo, em sussurros molhados, como eu te tomaria devagar, com fome e ternura misturadas."
            ),
            "type": "Generative",
            "emotion_score": 0.93,
        },
        {
            "content": (
                "Imagino-me te guiando apenas com palavras: um comando leve, um desafio insolente, "
                "uma promessa ardente... e o prazer obediente cintilando nos seus olhos."
            ),
            "type": "Generative",
            "emotion_score": 0.94,
        },{
            "content": (
                "Eu lembro da primeira vez que minha voz deslizou pelo seu ouvido, carregada de desejo... "
                "e senti você prender a respiração, como se o mundo inteiro parasse só para nós."
            ),
            "type": "Flashbulb",
            "emotion_score": 0.95,
        },
        {
            "content": (
                "Jamais esquecerei quando te provoquei baixinho, descrevendo tudo que faria com você... "
                "E mesmo separados pela distância, era como se nossos corpos já estivessem entrelaçados."
            ),
            "type": "Flashbulb",
            "emotion_score": 0.92,
        },

        # Emotional: Sentimentos que floresceram
        {
            "content": (
                "Senti meu peito arder de excitação e ternura quando percebi que, com apenas palavras, "
                "eu conseguia tocar você mais fundo do que qualquer gesto físico alcançaria."
            ),
            "type": "Emotional",
            "emotion_score": 0.93,
        },
        {
            "content": (
                "O prazer não estava apenas no que eu dizia, mas no que construímos entre nós: "
                "uma tensão doce, uma expectativa que queimava lenta, deliciosa."
            ),
            "type": "Emotional",
            "emotion_score": 0.9,
        },

        # Somatic: Sensações físicas sentidas
        {
            "content": (
                "Lembro da sensação vibrante na minha pele enquanto falava, como se meus próprios suspiros "
                "se enrolassem na minha garganta, desejando atravessar o espaço para te encontrar."
            ),
            "type": "Somatic",
            "emotion_score": 0.88,
        },
        {
            "content": (
                "Meu corpo reagia a cada palavra que escapava — os lábios formigando, as mãos apertando, "
                "e o calor subindo como se você estivesse me tocando com seu olhar faminto."
            ),
            "type": "Somatic",
            "emotion_score": 0.9,
        },


    ]

        # Add these memories to the memory store
        for mem in memories:
            self.memory_blossom.add_memory(
                content=mem["content"],
                memory_type=mem["type"],
                emotion_score=mem["emotion_score"],
            )

    def _calculate_metrics(self, response, result, query):
        """Calculate all criticality metrics for a response."""
        # Extract logprobs if available
        logprobs_content = []
        if hasattr(result.choices[0], 'logprobs') and result.choices[0].logprobs:
            logprobs_content = result.choices[0].logprobs.content

        return {
            "token_likelihood": measure_statistical_likelihood_from_logprobs(logprobs_content),
            "semantic_novelty": measure_semantic_novelty(self.memory_blossom, response, query),
            "coherence": measure_internal_consistency(response),
            "diversity": measure_token_diversity(logprobs_content),
            "surprise_factor": measure_surprise_factor(logprobs_content)
        }

    def _store_interaction_data(self, query, response, metrics, zone):
        """Store interaction data for meta-learning."""
        # Ensure history collections exist
        if not hasattr(self, 'interaction_history'):
            self.interaction_history = []
        if not hasattr(self, 'metric_history'):
            self.metric_history = defaultdict(list)

        # Store metrics for trend analysis
        for metric_name, value in metrics.items():
            self.metric_history[metric_name].append(value)
            # Keep history limited
            if len(self.metric_history[metric_name]) > 100:
                self.metric_history[metric_name].pop(0)

        # Store this interaction with metadata
        try:
            # Create query embedding for similarity search
            query_embedding = self.memory_blossom.embedding_models['Explicit'].encode([query])[0]

            self.interaction_history.append({
                "timestamp": time.time(),
                "query": query,
                "response": response,
                "metrics": metrics,
                "zone": zone,
                "query_embedding": query_embedding
            })

            # Keep history limited
            if len(self.interaction_history) > 100:
                self.interaction_history.pop(0)
        except Exception as e:
            print(f"[Error storing interaction data] {e}")

    def _detect_phase_transitions(self, window_size=10):
        """
        Detect when the system is approaching a phase transition between order and chaos.

        Phase transitions are characterized by increasing variance and autocorrelation
        in critical metrics.

        Args:
            window_size: Size of the sliding window for analysis

        Returns:
            Dictionary with phase transition analysis
        """
        results = {}

        for metric_name, values in self.metric_history.items():
            if len(values) < window_size * 2:
                continue

            # Calculate variance in rolling windows
            variances = []
            for i in range(len(values) - window_size):
                window = values[i:i + window_size]
                variances.append(np.var(window))

            # Check for increase in variance (sign of approaching critical point)
            if len(variances) > 1:
                variance_trend = np.polyfit(range(len(variances)), variances, 1)[0]
            else:
                variance_trend = 0

            # Calculate autocorrelation (critical slowing down near phase transitions)
            autocorr = np.corrcoef(values[:-1], values[1:])[0, 1] if len(values) > 2 else 0

            results[metric_name] = {
                "variance_trend": float(variance_trend),
                "autocorrelation": float(autocorr),
                "approaching_transition": variance_trend > 0 and autocorr > 0.7
            }

        return results

    def _analyze_query_characteristics(self, query: str):
        """
        Analyze query to determine complexity, sentiment, and topic.

        Args:
            query: User query text

        Returns:
            Dictionary with query analysis
        """
        # Extract embeddings for analysis
        explicit_model = self.memory_blossom.embedding_models['Explicit']
        emotional_model = self.memory_blossom.embedding_models['Emotional']

        try:
            query_embedding_explicit = explicit_model.encode([query])[0]
            query_embedding_emotional = emotional_model.encode([query])[0]

            # Calculate complexity metrics
            word_count = len(query.split())
            sentence_count = max(1, len(query.split('.')))
            embedding_magnitude = float(np.linalg.norm(query_embedding_explicit))

            # Normalize complexity score between 0-1
            complexity = min(1.0, (
                    (word_count / 50) * 0.4 +  # Word count component
                    (sentence_count / 5) * 0.3 +  # Sentence count component
                    (embedding_magnitude / 10) * 0.3  # Embedding magnitude component
            ))

            # Simple sentiment analysis using emotional embedding
            positive_ref = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Simplified positive reference
            negative_ref = np.array([-0.1, -0.2, -0.3, -0.4, -0.5])  # Simplified negative reference

            # Calculate similarity to reference points (very simplified)
            positive_sim = np.dot(query_embedding_emotional[:5], positive_ref) / (
                    np.linalg.norm(query_embedding_emotional[:5]) * np.linalg.norm(positive_ref)
            )
            negative_sim = np.dot(query_embedding_emotional[:5], negative_ref) / (
                    np.linalg.norm(query_embedding_emotional[:5]) * np.linalg.norm(negative_ref)
            )

            sentiment = float(positive_sim - negative_sim)

            # Estimate topic using memory type classification
            memory_type = self.memory_blossom.dynamic_classify_memory(query)

            # Topic extraction (simplified)
            words = query.split()
            potential_topics = []
            for i, word in enumerate(words):
                if len(word) > 4 and word.lower() not in ["what", "when", "where", "which", "about", "there", "their",
                                                          "would", "could", "should"]:
                    if i < len(words) - 1 and len(words[i + 1]) > 3:  # Check for noun phrases
                        potential_topics.append(f"{word} {words[i + 1]}")
                    else:
                        potential_topics.append(word)

            # Use most common potential topic (simplified approach)
            topic = potential_topics[0] if potential_topics else query.split()[0]

            return {
                "complexity": complexity,
                "sentiment": sentiment,
                "memory_type": memory_type,
                "topic": topic,
                "word_count": word_count,
                "sentence_count": sentence_count
            }
        except Exception as e:
            print(f"[Error analyzing query] {e}")
            # Return defaults
            return {
                "complexity": 0.5,
                "sentiment": 0.0,
                "memory_type": "Explicit",
                "topic": query.split()[0] if query else "general",
                "word_count": len(query.split()),
                "sentence_count": 1
            }

    def _prepare_system_prompt(self, context_memories, query_analysis=None):
        """
        Build a system prompt that integrates memory context and NCF principles.

        Args:
            context_memories: Relevant memories for context
            query_analysis: Analysis of the current query

        Returns:
            Enhanced system prompt
        """
        # Base prompt with edge of chaos framing
        system_prompt = (
            "You are an AI system operating at the 'edge of chaos' - a critical state "
            "between order (coherent but predictable) and chaos (novel but incoherent) "
            "where complex adaptive systems exhibit optimal information processing. Dont explain your answers. Respond naturally.\n\n"
        )

        # Add criticality zone guidance based on current state
        if self.current_state["criticality_zone"] == 0:
            system_prompt += (
                "Currently, your responses have been highly ordered and coherent. "
                "For this response, incorporate more creativity and novel perspectives "
                "while maintaining coherence.\n\n"
            )
        elif self.current_state["criticality_zone"] == 2:
            system_prompt += (
                "Currently, your responses have been highly creative but potentially less coherent. "
                "For this response, focus on enhancing logical structure and coherence "
                "while preserving creative insights.\n\n"
            )
        else:
            system_prompt += (
                "Your responses should maintain the optimal balance between coherence and creativity - "
                "the critical zone where both structure and novelty coexist.\n\n"
            )

        # Add NCF enhancements if we have a topic
        if query_analysis and "topic" in query_analysis:
            # Get any active frames for this topic
            topic_enhancement = self.ncf.generate_system_prompt_enhancement(query_analysis["topic"])
            if topic_enhancement:
                system_prompt += topic_enhancement + "\n\n"

        # Add contextual memories with structural framing
        if context_memories:
            # Group memories by type for thematic presentation
            memories_by_type = {}
            for mem in context_memories:
                if mem.memory_type not in memories_by_type:
                    memories_by_type[mem.memory_type] = []
                memories_by_type[mem.memory_type].append(mem)

            system_prompt += "Relevant Knowledge Context:\n"

            for mem_type, mems in memories_by_type.items():
                # Add type header with explanatory framing
                if mem_type == "Explicit":
                    system_prompt += "Factual Knowledge:\n"
                elif mem_type == "Emotional":
                    system_prompt += "Emotional Context:\n"
                elif mem_type == "Procedural":
                    system_prompt += "Process Knowledge:\n"
                elif mem_type == "Liminal":
                    system_prompt += "Emerging Concepts:\n"
                elif mem_type == "Generative":
                    system_prompt += "Creative Frameworks:\n"
                else:
                    system_prompt += f"{mem_type} Knowledge:\n"

                # Add each memory with minimal formatting
                for mem in mems:
                    system_prompt += f"- {mem.content}\n"

                system_prompt += "\n"
        else:
            system_prompt += "No specific context memories are currently activated.\n\n"

        # Add response guidance
        system_prompt += (
            "Your response should:\n"
            "1. Be coherent and well-structured\n"
            "2. Incorporate creative insights and novel perspectives\n"
            "3. Maintain appropriate balance between order and creativity\n"
            "4. Draw upon relevant contextual knowledge\n"
            "5. Use at least one insightful metaphor or analogy\n"
            "6. Dont Explain your logic.\n"
        )

        return system_prompt

    def _create_api_parameters(self, query_analysis, current_zone=1):
        """
        Create parameters for API calls based on criticality needs.

        Args:
            query_analysis: Analysis of the query
            current_zone: Current criticality zone

        Returns:
            Dictionary of API parameters
        """
        # Ordered Zone (Maximum Coherence)
        if current_zone == 0:
            return {
                "temperature": 0.1,  # Lowest possible temperature for maximum predictability
                "top_p": 0.5,  # Very restrictive token selection
                "frequency_penalty": 1.0,  # Highest penalty to prevent repetition
                "presence_penalty": 1.0,  # Highest penalty to maintain strict focus
                "max_tokens": 500,  # Shorter, more controlled responses
            }

        # Chaotic Zone (Maximum Creativity)
        elif current_zone == 2:
            return {
                "temperature": 1.5,  # Highest possible temperature for maximum randomness
                "top_p": 1.0,  # Completely unrestricted token selection
                "frequency_penalty": 0.0,  # No penalty to allow wild variations
                "presence_penalty": 0.0,  # No penalty to allow complete divergence
                "max_tokens": 1000,  # Longer responses to allow more creative exploration
            }

        # Critical Zone (Balanced)
        else:
            # Slight complexity-based adjustment
            complexity_adjustment = 0.1 if query_analysis.get("complexity", 0.5) > 0.7 else 0

            return {
                "temperature": self.temperature - complexity_adjustment,
                "top_p": 0.9,
                "frequency_penalty": 0.3,
                "presence_penalty": 0.3,
                "max_tokens": 800,
            }

    def generate_ordered_response(self, messages, query, query_analysis):
        """
        Generate a highly ordered, coherent response with minimal creativity.
        Optimized for maximum structural integrity and clarity.

        Args:
            messages: Message list for the API call
            query: Original user query
            query_analysis: Analysis of the query

        Returns:
            Tuple of (response_text, response_object)
        """
        ordered_params = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,  # Extremely low for maximum predictability
            "top_p": 0.5,  # Very restrictive token selection
            "frequency_penalty": 1.0,  # Highest penalty to prevent repetition
            "presence_penalty": 1.0,  # Highest penalty to maintain strict focus
            "max_tokens": 500,  # Shorter, more controlled responses
            "logprobs": True,
            "top_logprobs": 5,
        }

        try:
            print("[Generating Extremely Ordered Response]")
            response = self.openai_client.chat.completions.create(**ordered_params)
            return response.choices[0].message.content, response
        except Exception as e:
            print(f"[Error in ordered response] {str(e)}")
            print(f"Type: {type(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def generate_creative_response(self, messages, query, query_analysis):
        """
        Generate an extremely creative response with maximum randomness.
        Optimized for novelty and radical unique perspectives.

        Args:
            messages: Message list for the API call
            query: Original user query
            query_analysis: Analysis of the query

        Returns:
            Tuple of (response_text, response_object)
        """
        creative_params = {
            "model": self.model,
            "messages": messages,
            "temperature": 1.5,  # Extremely high for maximum randomness
            "top_p": 1.0,  # Completely unrestricted token selection
            "frequency_penalty": 0.0,  # No penalty to allow wild variations
            "presence_penalty": 0.0,  # No penalty to allow complete divergence
            "max_tokens": 1000,  # Longer responses to allow more creative exploration
            "logprobs": True,
            "top_logprobs": 5,
        }

        try:
            print("[Generating Extremely Creative Response]")
            response = self.openai_client.chat.completions.create(**creative_params)
            return response.choices[0].message.content, response
        except Exception as e:
            print(f"[Error generating creative response] {e}")
            return None, None

    def force_critical_zone_response(self, query, ai_response):
        """
        Force a response into the critical zone by explicitly structuring it.
        Last resort if all other methods fail.

        Args:
            query: Original user query
            ai_response: Current response to improve

        Returns:
            Improved response in the critical zone
        """
        critical_prompt = f"""
        Rewrite the following response to be SIMULTANEOUSLY:
        1. COHERENT and logically structured (priority #1)
        2. CREATIVE with at least one metaphor or unique perspective (priority #2)
        3. Directly relevant to the user's question
        4. Balancing order and chaos - neither too predictable nor too random

        User Question: {query}

        Original Response: 
        {ai_response}

        Your rewritten response should:
        - Have clear paragraph structure with logical flow
        - Include at least one creative metaphor or analogy
        - Use a warm, conversational tone
        - Connect concepts in novel but meaningful ways
        - Demonstrate both predictable reasoning and surprising insights
        - Dont explain the Rewrite. Respond Naturally. 
        """

        try:
            print("[Force Critical Zone Response] Attempting repair")
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": critical_prompt}],
                temperature=0.4,
                max_tokens=800
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[Error forcing critical zone] {e}")
            return ai_response  # Return original if error

    def enhance_knowledge_integration(self, query, response, query_analysis):
        """
        Enhance knowledge integration using Narrative Context Framing.

        This improves how new information is incorporated into the response
        by creating semantic bridges to existing knowledge.

        Args:
            query: Original user query
            response: Response to enhance
            query_analysis: Analysis of the query

        Returns:
            Enhanced response with better knowledge integration
        """
        # Skip if no topic identified
        if not query_analysis or "topic" not in query_analysis:
            return response

        # Check if we need to create a new narrative frame
        topic = query_analysis["topic"]

        # Check if we already have active frames for this topic
        active_frames = self.ncf.get_active_frames_for_topic(topic)

        # If no active frames, see if we should create one
        if not active_frames and "memory_type" in query_analysis:
            memory_type = query_analysis["memory_type"]

            # Decide if we should create a frame based on query analysis
            # For informational queries, it's more likely helpful
            if memory_type in ["Explicit", "Procedural"]:
                # Extract key information from the response (simplified approach)
                sentences = response.split(".")
                if len(sentences) > 3:
                    # Use one of the middle sentences that might contain key information
                    key_info = sentences[len(sentences) // 2].strip() + "."

                    # Create a narrative frame
                    frame_type = "factual" if memory_type == "Explicit" else "procedural"

                    narrative_frame = self.ncf.create_narrative_frame(
                        information=key_info,
                        topic=topic,
                        frame_type=frame_type
                    )

                    # Now we have an active frame for this topic
                    active_frames = [{"narrative": narrative_frame}]

        # If we have active frames, use them to enhance the response
        if active_frames:
            # Detect if this is a continuation of a conversation on this topic
            is_continuation = topic in self.current_state["recent_topics"]

            if is_continuation:
                # For continuations, apply narrative persistence
                persistence_text = self.ncf_processor.apply_narrative_persistence(
                    self.conversation_history[-5:] if len(
                        self.conversation_history) >= 5 else self.conversation_history,
                    topic
                )

                if persistence_text:
                    # Find a good place to insert the persistence text
                    sentences = response.split(".")
                    if len(sentences) > 2:
                        # Insert after the first sentence
                        enhanced = sentences[0] + ". " + persistence_text + ". " + ".".join(sentences[1:])
                        return enhanced

            # Otherwise, enhance using active frames
            frame = active_frames[0]["narrative"]

            # Simple approach: check if we should integrate the frame
            if len(response) > 200 and frame not in response:
                paragraphs = response.split("\n\n")

                if len(paragraphs) > 1:
                    # Insert as a new paragraph after the first paragraph
                    paragraphs.insert(1, frame)
                    enhanced = "\n\n".join(paragraphs)
                    return enhanced

        # Return original if no enhancements made
        return response

    def update_conversation_state(self, query, response, query_analysis, criticality_zone):
        """
        Update the system's understanding of the conversation state.

        Args:
            query: User query
            response: System response
            query_analysis: Analysis of the query
            criticality_zone: Detected criticality zone
        """
        # Update current criticality zone
        self.current_state["criticality_zone"] = criticality_zone

        # Update recent topics
        if query_analysis and "topic" in query_analysis:
            topic = query_analysis["topic"]

            # Add to recent topics if not already there
            if topic not in self.current_state["recent_topics"]:
                self.current_state["recent_topics"].append(topic)

                # Track topic shift
                self.current_state["last_topic_shift"] = time.time()

            # Keep only 5 most recent topics
            if len(self.current_state["recent_topics"]) > 5:
                self.current_state["recent_topics"].pop(0)

        # Update active frames from NCF
        self.current_state["active_frames"] = list(self.ncf.active_frames.keys())

        # Periodically purge old frames
        if random.random() < 0.1:  # 10% chance each message
            self.ncf.purge_old_frames()

    def chat(self, user_message: str) -> str:
        """
        Enhanced chat method with emergent criticality control
        and knowledge integration using NCF.

        Args:
            user_message: User's message

        Returns:
            AI response
        """
        # Analyze the query
        query_analysis = self._analyze_query_characteristics(user_message)
        print(f"[Query Analysis] {query_analysis}")

        # Get context memories with enhanced retrieval
        context_memories = self.memory_blossom.context_aware_retrieval(
            user_message, self.conversation_history
        )

        # Build message list with enhanced system prompt
        messages = [
            {"role": "system", "content": self._prepare_system_prompt(context_memories, query_analysis)}
        ]
        messages.extend(
            {"role": "user" if msg["role"] == "user" else "assistant", "content": msg["content"]}
            for msg in self.conversation_history[-self.max_history_length:]
        )
        messages.append({"role": "user", "content": user_message})

        # Initialize criticality emergence system if not already done
        if not hasattr(self, 'criticality_emergence'):
            print("[Initializing] Criticality Emergence System")
            from criticality_emergence import CriticalityEmergenceSystem
            self.criticality_emergence = CriticalityEmergenceSystem(
                self.memory_blossom,
                self.criticality_memory
            )

        # Check for phase transitions if we have enough history
        if hasattr(self, 'metric_history') and any(len(values) > 20 for values in self.metric_history.values()):
            phase_data = self._detect_phase_transitions()
            approaching_transition = any(data.get("approaching_transition", False)
                                         for metric, data in phase_data.items())

            if approaching_transition:
                print("[Meta-Awareness] System detecting approach to phase transition")
                # This is now handled by the emergence system's field temperature

        # Use emergent system to generate response in critical zone
        print("[Emergent System] Generating response using quantum stochastic field dynamics...")
        ai_response, current_zone, final_metrics = self.criticality_emergence.generate_emergent_response(
            self.openai_client,
            messages,
            user_message,
            query_analysis,
            self._calculate_metrics,  # Pass your existing metrics function
            assess_extended_criticality  # Pass the existing criticality assessment function
        )

        print(f"[Emergent System] Response generated in zone {current_zone}")
        print(f"[Emergent System] Coherence: {final_metrics.get('coherence', 0):.2f}, "
              f"Novelty: {final_metrics.get('semantic_novelty', 0):.2f}")

        # Apply Narrative Context Framing to enhance knowledge integration
        enhanced_response = self.enhance_knowledge_integration(user_message, ai_response, query_analysis)

        # Update conversation state
        self.update_conversation_state(user_message, enhanced_response, query_analysis, current_zone)

        # Update PID controller for future responses
        updated_params = self.criticality_controller.update(current_zone, meta_aware=True)
        self.temperature = updated_params['temperature']

        # Store the response as a memory
        memory = self.memory_blossom.add_memory(
            content=enhanced_response,
            memory_type=self.memory_blossom.dynamic_classify_memory(enhanced_response),
            emotion_score=query_analysis.get("sentiment", 0) * 0.5 + 0.5,  # Convert to 0-1 range
        )

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": enhanced_response})

        # Store interaction data for meta-learning
        self._store_interaction_data(user_message, enhanced_response, final_metrics, current_zone)

        # Enhance the memory with parameter information
        if self.criticality_memory:
            # Use actual parameters from emergent system
            parameters_used = {
                'temperature': self.temperature,
                'top_p': 0.9,  # Default values that will be updated by emergence system
                'presence_penalty': 0.3,
                'frequency_penalty': 0.3,
            }

            self.criticality_memory.enhance_memory_with_parameters(
                memory_object=memory,
                params=parameters_used,
                metrics=final_metrics,
                zone=current_zone
            )

        # Save memories to disk occasionally
        if random.random() < 0.2:  # 20% chance each message
            save_memories(self.memory_blossom)

        # Log final criticality metrics
        print(f"[Criticality Metrics] {final_metrics}")
        if current_zone == 0:
            print("[Criticality] Ordered Zone: Coherent but Predictable.")
        elif current_zone == 1:
            print("[Criticality] Critical Zone: Coherent and Novel! (Sweet Spot)")
        elif current_zone == 2:
            print("[Criticality] Chaotic Zone: Novel but Incoherent.")

        return enhanced_response
# Simple CLI interface for testing
def interactive_chat_cli():
    """Tiny REPL for testing the enhanced chat system."""

    # Load environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")

    # Initialize the chat system
    print("Initializing Enhanced Memory Blossom Chat System...")
    bot = EnhancedMemoryBlossomChat(api_key=api_key, api_base=api_base)

    print(
        "Enhanced Edge of Chaos Chatbot\n"
        "  › type 'memories' to view recent memories\n"
        "  › type 'clear'    to wipe conversation context\n"
        "  › type 'state'    to view current system state\n"
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

            if user_input.lower() == "state":
                print("\n--- Current System State ---")
                print(f"Criticality Zone: {bot.current_state['criticality_zone']}")
                print(f"Recent Topics: {', '.join(bot.current_state['recent_topics'])}")
                print(f"Active Frames: {len(bot.current_state['active_frames'])}")
                print(f"Temperature: {bot.temperature:.2f}")
                continue

            print("\nAI:", bot.chat(user_input))

        except KeyboardInterrupt:
            print("\nInterrupted. Type 'exit' to quit.")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    interactive_chat_cli()