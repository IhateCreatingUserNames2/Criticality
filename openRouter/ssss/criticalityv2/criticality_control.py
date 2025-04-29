"""
Criticality Control System

This module implements the "Edge of Chaos" theory for AI responses,
maintaining the system in the critical zone between order and chaos
where optimal information processing occurs.

Key concepts:
1. Order - Highly coherent, predictable, but potentially uninteresting responses
2. Chaos - Highly novel, creative, but potentially incoherent responses
3. Criticality - The optimal balance point between order and chaos
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity


class CriticalityController:
    """
    PID Controller for maintaining the system in the critical zone.
    Dynamically adjusts parameters based on the current and target criticality.
    """

    def __init__(self):
        # PID controller parameters
        self.kp = 0.7  # Proportional gain
        self.ki = 0.15  # Integral gain
        self.kd = 0.2  # Derivative gain

        # State variables
        self.target = 1.0  # Target is critical zone (1)
        self.previous_error = 0
        self.integral = 0
        self.error_history = []

        # Current parameters
        self.temperature = 0.7
        self.top_p = 0.9
        self.presence_penalty = 0.3
        self.frequency_penalty = 0.3

        # Meta-awareness variables
        self.adaptation_rate = 0.05
        self.successful_zones = []
        self.parameter_effectiveness = {
            "temperature": 0.0,
            "top_p": 0.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0
        }

        # Learning rate for parameter adaptation
        self.learning_rate = 0.05

        # Phase transition detection
        self.transition_detection_enabled = True
        self.transition_window_size = 10  # How many samples to consider
        self.transition_threshold = 0.2  # Threshold for detecting transitions

    def update(self, current_zone: int, meta_aware: bool = False) -> Dict[str, float]:
        """
        Update parameters based on current criticality zone using PID control.

        Args:
            current_zone: Current criticality zone (0=ordered, 1=critical, 2=chaotic)
            meta_aware: Whether to use meta-awareness to adjust parameters

        Returns:
            Dictionary with updated parameters
        """
        # Map zones to numerical values for PID controller
        zone_values = {0: 0.0, 1: 1.0, 2: 2.0}
        current_value = zone_values[current_zone]

        # Calculate error (difference from target)
        error = self.target - current_value

        # Track error history for debugging and meta-learning
        self.error_history.append(error)
        if len(self.error_history) > 20:  # Keep more history for meta-learning
            self.error_history.pop(0)

        # Calculate PID components
        self.integral = max(-3.0, min(3.0, self.integral + error))  # Limit integral windup
        derivative = error - self.previous_error

        # Calculate adjustment using PID formula
        adjustment = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        # Store previous parameters for meta-learning
        prev_params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty
        }

        # Update parameters with limits
        self.temperature = max(0.1, min(1.5, self.temperature + (adjustment * 0.2)))
        self.top_p = max(0.5, min(0.98, self.top_p + (adjustment * 0.1)))
        self.frequency_penalty = max(0.0, min(1.0, self.frequency_penalty + (adjustment * 0.1)))
        self.presence_penalty = max(0.0, min(1.0, self.presence_penalty + (adjustment * 0.1)))

        # Meta-learning: adjust PID parameters based on effectiveness
        if meta_aware and len(self.error_history) >= 5:
            self._adapt_control_parameters()

            # Store successful parameter combinations
            if current_zone == 1:  # We're in the critical zone
                self.successful_zones.append(prev_params)
                if len(self.successful_zones) > 10:
                    self.successful_zones.pop(0)

        # Update state
        self.previous_error = error

        # Log for debugging
        print(f"[PID Controller] Error: {error:.2f}, Adjustment: {adjustment:.2f}")
        print(f"[PID Controller] New params: temp={self.temperature:.2f}, top_p={self.top_p:.2f}")

        # Check for phase transitions
        if meta_aware and self.transition_detection_enabled:
            transition_detected = self._detect_phase_transition()
            if transition_detected:
                print("[Meta-Controller] Phase transition detected - applying parameter correction")
                self._apply_transition_correction()

        # Use meta-learning to consider previous successful parameters
        if meta_aware and len(self.successful_zones) > 3 and current_zone != 1:
            return self._apply_meta_learning_correction(current_zone)

        return {
            'temperature': self.temperature,
            'top_p': self.top_p,
            'frequency_penalty': self.frequency_penalty,
            'presence_penalty': self.presence_penalty
        }

    def _adapt_control_parameters(self):
        """Adapt PID control parameters based on recent error history."""
        # Error trend indicates whether we're approaching or moving away from target
        recent_errors = self.error_history[-5:]
        error_trend = np.polyfit(range(len(recent_errors)), recent_errors, 1)[0]

        if error_trend < 0:  # Error is decreasing - good direction
            # Reward by increasing adaptation rate
            self.adaptation_rate += 0.01

            # Gradually increase proportional gain if it's helping
            self.kp = self.kp * (1 + self.learning_rate * 0.1)
        else:  # Error is increasing or flat - wrong direction
            # Reduce adaptation rate
            self.adaptation_rate = max(0.01, self.adaptation_rate - 0.01)

            # Adjust PID parameters
            self.kp = self.kp * (1 - self.learning_rate * 0.1)  # Reduce proportional gain
            self.ki = max(0.05, self.ki * (1 - self.learning_rate * 0.05))  # Reduce integral gain

        # Ensure parameter bounds
        self.kp = max(0.1, min(2.0, self.kp))
        self.ki = max(0.01, min(0.5, self.ki))
        self.kd = max(0.05, min(1.0, self.kd))

    def _detect_phase_transition(self) -> bool:
        """
        Detect when the system is approaching a phase transition.

        Phase transitions are characterized by increasing variance and
        autocorrelation in error measurements.

        Returns:
            True if a phase transition is detected
        """
        if len(self.error_history) < self.transition_window_size * 2:
            return False

        # Variance trend (increasing variance suggests approach to transition)
        variances = []
        for i in range(len(self.error_history) - self.transition_window_size):
            window = self.error_history[i:i + self.transition_window_size]
            variances.append(np.var(window))

        if len(variances) < 2:
            return False

        # Calculate variance trend using linear regression
        variance_trend = np.polyfit(range(len(variances)), variances, 1)[0]

        # Autocorrelation (critical slowing down near transitions)
        autocorr = np.corrcoef(self.error_history[:-1], self.error_history[1:])[0, 1]

        # Detect transition
        approaching_transition = (
                variance_trend > self.transition_threshold and
                autocorr > 0.7
        )

        return approaching_transition

    def _apply_transition_correction(self):
        """Apply correction when approaching a phase transition."""
        # When approaching transition, make smaller adjustments
        self.kp *= 0.8  # Reduce proportional gain
        self.ki *= 0.5  # Reduce integral gain
        self.kd *= 1.5  # Increase derivative gain (dampening)

        # Reset integral term to prevent overshooting
        self.integral = 0

    def _apply_meta_learning_correction(self, current_zone: int) -> Dict[str, float]:
        """
        Apply correction based on previously successful parameters.

        Args:
            current_zone: Current criticality zone

        Returns:
            Updated parameters dictionary
        """
        print("[Meta-Controller] Considering previous successful parameters")

        # Calculate average of successful parameters
        avg_successful = {
            param: sum(zone[param] for zone in self.successful_zones) / len(self.successful_zones)
            for param in ["temperature", "top_p", "presence_penalty", "frequency_penalty"]
        }

        # Blend current with successful (80% successful, 20% current) - adaptive blending
        # Use more of successful parameters when far from critical zone
        blend_factor = 0.9 if current_zone == 0 else 0.7  # Use more correction in ordered zone

        self.temperature = blend_factor * avg_successful["temperature"] + (1 - blend_factor) * self.temperature
        self.top_p = blend_factor * avg_successful["top_p"] + (1 - blend_factor) * self.top_p
        self.presence_penalty = blend_factor * avg_successful["presence_penalty"] + (
                    1 - blend_factor) * self.presence_penalty
        self.frequency_penalty = blend_factor * avg_successful["frequency_penalty"] + (
                    1 - blend_factor) * self.frequency_penalty

        print(f"[Meta-Controller] Blended with successful params: temp={self.temperature:.2f}, top_p={self.top_p:.2f}")

        return {
            'temperature': self.temperature,
            'top_p': self.top_p,
            'frequency_penalty': self.frequency_penalty,
            'presence_penalty': self.presence_penalty
        }


class CriticalityMemory:
    """
    Memory system that remembers successful parameter combinations for different query types.
    This enables the system to learn what parameters work best for different types of queries,
    implementing a form of meta-learning at the edge of chaos.
    """

    def __init__(self, embedding_model=None):
        self.successful_params = []
        self.max_memories = 50  # Increased from 30 for more history
        self.embedding_model = embedding_model

        # Parameter performance tracking
        self.parameter_performance = defaultdict(list)  # {param_name: [(value, performance)]}

        # Phase space tracking (map different regions of the parameter space)
        self.phase_space = {
            'ordered_region': [],  # Parameter sets that led to ordered responses
            'critical_region': [],  # Parameter sets that led to critical responses
            'chaotic_region': []  # Parameter sets that led to chaotic responses
        }

        # Memory decay parameters
        self.memory_decay_rate = 0.01
        self.last_decay_time = time.time()

    def record_success(self, query: str, params: Dict[str, Any], metrics: Dict[str, float], zone: int = 1):
        """
        Record parameters that produced a response in the critical zone.

        Args:
            query: The user query that produced a successful response
            params: The parameters used
            metrics: The criticality metrics that were measured
            zone: The criticality zone (0=ordered, 1=critical, 2=chaotic)
        """
        if len(self.successful_params) >= self.max_memories:
            self.successful_params.pop(0)  # Remove oldest

        # Create embedding if available
        query_embedding = None
        if self.embedding_model:
            try:
                query_embedding = self.embedding_model.encode([query])[0]
            except Exception as e:
                print(f"[Criticality Memory] Error creating embedding: {e}")

        # Record the memory
        memory = {
            'query': query,
            'query_embedding': query_embedding,
            'params': params,
            'metrics': metrics,
            'zone': zone,
            'timestamp': time.time(),
            'performance_score': self._calculate_performance_score(metrics, zone)
        }

        self.successful_params.append(memory)

        # Update phase space
        if zone == 0:
            self.phase_space['ordered_region'].append(params)
        elif zone == 1:
            self.phase_space['critical_region'].append(params)
        elif zone == 2:
            self.phase_space['chaotic_region'].append(params)

        # Keep phase space regions limited
        for region in self.phase_space:
            if len(self.phase_space[region]) > 20:
                self.phase_space[region].pop(0)

        # Update parameter performance tracking
        for param_name, value in params.items():
            self.parameter_performance[param_name].append(
                (value, memory['performance_score'])
            )

            # Keep performance tracking limited
            if len(self.parameter_performance[param_name]) > 100:
                self.parameter_performance[param_name].pop(0)

        print(f"[Criticality Memory] Recorded parameters for zone {zone} query: {query[:30]}...")

        # Periodically apply memory decay
        current_time = time.time()
        if current_time - self.last_decay_time > 3600:  # Every hour
            self._apply_memory_decay()
            self.last_decay_time = current_time

    def enhance_memory_with_parameters(self, memory_object, params, metrics, zone):
        """
        Enhance a memory object with the parameters that produced it.

        Args:
            memory_object: The Memory object to enhance
            params: The parameters used to generate this memory
            metrics: The criticality metrics measured
            zone: The criticality zone (0=ordered, 1=critical, 2=chaotic)
        """
        # Add parameter information to memory metadata
        if not hasattr(memory_object, 'metadata') or memory_object.metadata is None:
            memory_object.metadata = {}

        # Store parameters that produced this memory
        memory_object.metadata['generation_params'] = params.copy()
        memory_object.metadata['criticality_metrics'] = metrics.copy()
        memory_object.metadata['criticality_zone'] = zone
        memory_object.metadata['parameter_performance'] = self._calculate_performance_score(metrics, zone)

        # Add timestamp for when these parameters were used
        memory_object.metadata['parameter_timestamp'] = time.time()

        print(f"[Memory Enhancement] Memory enhanced with generation parameters (zone {zone})")
        return memory_object
    def _calculate_performance_score(self, metrics: Dict[str, float], zone: int) -> float:
        """
        Calculate a performance score for the parameter set.

        Higher scores indicate better performance.

        Args:
            metrics: Criticality metrics
            zone: Criticality zone achieved

        Returns:
            Performance score (0-1)
        """
        # Base score depends on whether we reached the critical zone
        base_score = 1.0 if zone == 1 else 0.5

        # Adjust based on coherence and creativity
        coherence = metrics.get('coherence', 0.5)
        novelty = metrics.get('semantic_novelty', 0.5)

        # Ideal balance is high coherence with moderate novelty
        balance_score = coherence * 0.6 + (1.0 - abs(novelty - 0.5)) * 0.4

        return base_score * 0.7 + balance_score * 0.3

    def _apply_memory_decay(self):
        """Apply decay to older memories to gradually forget outdated parameters."""
        current_time = time.time()

        for memory in self.successful_params:
            # Calculate age in hours
            age_hours = (current_time - memory['timestamp']) / 3600

            # Apply exponential decay to performance score
            decay_factor = np.exp(-self.memory_decay_rate * age_hours)
            memory['performance_score'] *= decay_factor

    def get_closest_params(self, query: str, current_zone: int = None) -> Optional[Dict[str, Any]]:
        """
        Return parameters most similar to the current context.

        Args:
            query: The current user query
            current_zone: The current criticality zone (if known)

        Returns:
            Dictionary of parameters or None if no memories exist
        """
        if not self.successful_params:
            return None

        # If we're already in the critical zone, no need to change
        if current_zone == 1:
            return None

        # If we have an embedding model, find the most similar query
        query_params = None
        if self.embedding_model:
            try:
                query_embedding = self.embedding_model.encode([query])[0]

                # Find the parameters most similar to the current context
                # with a focus on successful (critical zone) parameters
                similarities = []
                for entry in self.successful_params:
                    if entry['query_embedding'] is not None:
                        similarity = cosine_similarity([query_embedding], [entry['query_embedding']])[0][0]

                        # Weight by performance score and recency
                        age_weight = np.exp(-(time.time() - entry['timestamp']) / (3600 * 24 * 7))  # Week half-life

                        # Prioritize critical zone parameters (zone 1)
                        zone_weight = 1.5 if entry['zone'] == 1 else 1.0

                        weighted_similarity = similarity * entry['performance_score'] * age_weight * zone_weight
                        similarities.append((weighted_similarity, entry['params']))

                if similarities:
                    # Return the most similar with threshold
                    similarities.sort(key=lambda x: x[0], reverse=True)
                    top_match = similarities[0]

                    if top_match[0] > 0.6:  # Only use if similarity is significant
                        print(f"[Criticality Memory] Found similar query with score: {top_match[0]:.3f}")
                        query_params = top_match[1]
            except Exception as e:
                print(f"[Criticality Memory] Error computing similarity: {e}")

        # If no query-based parameters or similarity too low, use phase space knowledge
        if query_params is None and current_zone is not None:
            params = self._get_phase_space_recommendation(current_zone)
            if params:
                return params

        # Fallback to most recent successful parameters
        if not query_params and self.successful_params:
            # Find most recent critical zone parameters
            critical_params = [entry for entry in self.successful_params if entry['zone'] == 1]
            if critical_params:
                # Sort by recency
                critical_params.sort(key=lambda x: x['timestamp'], reverse=True)
                print("[Criticality Memory] Using most recent critical zone parameters")
                return critical_params[0]['params']

            # If no critical zone params, use most recent
            print("[Criticality Memory] Using most recent parameters")
            return self.successful_params[-1]['params']

        return query_params

    def _get_phase_space_recommendation(self, current_zone: int) -> Optional[Dict[str, Any]]:
        """
        Get a parameter recommendation based on phase space knowledge.

        This uses the system's understanding of parameter regions
        that tend to produce ordered, critical, or chaotic responses.

        Args:
            current_zone: Current criticality zone

        Returns:
            Parameter dictionary or None
        """
        # If currently in ordered zone (0), recommend parameters from critical region
        if current_zone == 0 and self.phase_space['critical_region']:
            # Use a parameter set from the critical region
            critical_params = self.phase_space['critical_region']

            # Choose randomly from top half of critical params (by recency)
            if len(critical_params) > 1:
                idx = np.random.randint(len(critical_params) // 2, len(critical_params))
                print("[Criticality Memory] Recommending parameters from critical region")
                return critical_params[idx]
            else:
                return critical_params[0]

        # If currently in chaotic zone (2), recommend parameters from critical region
        elif current_zone == 2 and self.phase_space['critical_region']:
            # Use a parameter set from the critical region
            critical_params = self.phase_space['critical_region']

            # Choose randomly from top half of critical params (by recency)
            if len(critical_params) > 1:
                idx = np.random.randint(len(critical_params) // 2, len(critical_params))
                print("[Criticality Memory] Recommending parameters from critical region")
                return critical_params[idx]
            else:
                return critical_params[0]

        return None

    def analyze_parameter_effectiveness(self) -> Dict[str, Any]:
        """
        Analyze which parameter values are most effective for reaching critical zone.

        Returns:
            Dictionary with parameter analysis
        """
        results = {}

        for param_name, values in self.parameter_performance.items():
            if len(values) < 10:
                continue

            # Extract values and scores
            param_values = [v[0] for v in values]
            scores = [v[1] for v in values]

            # Find average score for different value ranges
            bins = 5
            ranges = np.linspace(min(param_values), max(param_values), bins + 1)
            range_scores = []

            for i in range(bins):
                bin_start = ranges[i]
                bin_end = ranges[i + 1]

                # Get scores for values in this range
                bin_scores = [score for val, score in zip(param_values, scores)
                              if bin_start <= val <= bin_end]

                if bin_scores:
                    range_scores.append({
                        'range': (float(bin_start), float(bin_end)),
                        'avg_score': float(np.mean(bin_scores)),
                        'count': len(bin_scores)
                    })

            # Find optimal value range
            if range_scores:
                optimal_range = max(range_scores, key=lambda x: x['avg_score'])

                results[param_name] = {
                    'ranges': range_scores,
                    'optimal_range': optimal_range['range'],
                    'optimal_score': optimal_range['avg_score']
                }

        return results


def measure_token_diversity(logprobs_data):
    """
    Measures how diverse the token options were at each step.
    Higher diversity suggests more creative potential.

    This is a key metric for detecting the edge of chaos in language generation.

    Args:
        logprobs_data: Token logprobs from generation

    Returns:
        Diversity score (0-1)
    """
    if not logprobs_data or not hasattr(logprobs_data[0], 'top_logprobs'):
        return 0.5  # fallback to neutral if no data

    diversity_scores = []

    for token in logprobs_data:
        if hasattr(token, 'top_logprobs') and token.top_logprobs:
            # Get probabilities from logprobs
            probs = [np.exp(lp.logprob) for lp in token.top_logprobs]

            # Calculate entropy as diversity measure
            total = sum(probs)
            normalized_probs = [p / total for p in probs]
            entropy = -sum(p * np.log(p) for p in normalized_probs if p > 0)

            # Normalize to 0-1 range (assuming max entropy for 5 options is ~1.61)
            normalized_entropy = min(1.0, entropy / 1.61)
            diversity_scores.append(normalized_entropy)

    return np.mean(diversity_scores) if diversity_scores else 0.5


def measure_surprise_factor(logprobs_data):
    """
    Measures how often the selected token was not the most probable option.
    Higher values suggest more surprising/creative responses.

    Surprise is a key element of the edge of chaos - unexpected but meaningful choices.

    Args:
        logprobs_data: Token logprobs from generation

    Returns:
        Surprise score (0-1)
    """
    if not logprobs_data or not hasattr(logprobs_data[0], 'top_logprobs'):
        return 0.5  # fallback to neutral if no data

    surprise_count = 0
    total_tokens = 0

    # Track graded surprise (how surprising each token choice was)
    surprise_levels = []

    for token in logprobs_data:
        if hasattr(token, 'top_logprobs') and token.top_logprobs:
            total_tokens += 1

            # Check if selected token wasn't the most probable
            selected_logprob = token.logprob
            most_probable_logprob = max(lp.logprob for lp in token.top_logprobs)

            if selected_logprob < most_probable_logprob - 0.1:  # Small threshold for numerical stability
                surprise_count += 1

                # Calculate degree of surprise (how far down the probability ranking)
                surprise_degree = (most_probable_logprob - selected_logprob) / abs(most_probable_logprob)
                surprise_levels.append(min(1.0, surprise_degree))

    # Basic surprise factor
    basic_surprise = surprise_count / max(1, total_tokens)

    # If we have surprise levels, calculate weighted surprise
    if surprise_levels:
        weighted_surprise = sum(surprise_levels) / max(1, len(surprise_levels))
        # Combine basic and weighted surprise
        return (basic_surprise * 0.7 + weighted_surprise * 0.3)

    return basic_surprise


def measure_statistical_likelihood_from_logprobs(logprobs_data) -> float:
    """
    Measures how predictable (ordered) the model output is, based on real logprobs.
    Higher average logprobs = more confident = more ordered.

    This helps detect when the system is operating in the ordered regime.

    Args:
        logprobs_data: Token logprobs from generation

    Returns:
        Statistical likelihood score (0-1)
    """
    if not logprobs_data:
        print("[Warning] No logprobs received. Some criticality metrics defaulted to neutral values (0.5).")
        return 0.5  # fallback to neutral if no data

    # Extract logprob values
    logprob_values = [token.logprob for token in logprobs_data if token.logprob is not None]
    if not logprob_values:
        return 0.5

    # Basic average logprob
    avg_logprob = np.mean(logprob_values)

    # Calculate variance (higher variance suggests more complex/critical pattern)
    logprob_variance = np.var(logprob_values) if len(logprob_values) > 1 else 0

    # Normalize average: from (-inf, 0) to [0, 1]
    # Example: -5 very uncertain (~0.0), 0 is 100% confident (~1.0)
    normalized_avg = np.clip((avg_logprob + 5) / 5, 0, 1)

    # Normalize variance: higher variance = lower statistical likelihood
    normalized_variance = np.clip(logprob_variance / 2, 0, 1)
    variance_factor = 1.0 - normalized_variance * 0.3  # Discount likelihood somewhat for high variance

    # Combine into final score
    final_score = normalized_avg * variance_factor

    return final_score


def measure_semantic_novelty(memory_blossom, text: str, query: str, strategy: str = "worst") -> float:
    """
    Measure novelty using available embedding models.

    This detects how different (creative) the response is from the query.

    Args:
        memory_blossom: MemoryBlossom system
        text: Response text to evaluate
        query: Original query text
        strategy: "avg" (average similarity) or "worst" (minimum similarity)

    Returns:
        Novelty score (0-1)
    """
    from embedding_utils import compute_adaptive_similarity

    try:
        # Use multiple embedding models for robust novelty estimation
        models_to_try = ['Explicit', 'Emotional', 'Generative']
        similarities = []

        for model_name in models_to_try:
            if model_name in memory_blossom.embedding_models:
                model = memory_blossom.embedding_models[model_name]

                # Encode query and response
                text_embedding = model.encode([text])[0]
                query_embedding = model.encode([query])[0]

                # Calculate similarity using adaptive method
                similarity = compute_adaptive_similarity(text_embedding, query_embedding)
                similarities.append(similarity)

        # Ensure we have at least one valid similarity
        if not similarities:
            return 0.5  # Neutral if no valid similarities

        # Calculate final similarity based on strategy
        if strategy == "avg":
            final_similarity = sum(similarities) / len(similarities)
        elif strategy == "worst":
            final_similarity = min(similarities)
        else:
            final_similarity = similarities[0]  # Default to first

        # Convert similarity to novelty (inverse relationship)
        novelty = 1.0 - final_similarity

        return novelty
    except Exception as e:
        print(f"[Error measuring semantic novelty] {e}")
        return 0.5  # Fallback to neutral


def measure_internal_consistency(text: str) -> float:
    """
    Enhanced coherence check based on sentence structure and relationships.

    This detects how ordered and logical the response is.

    Args:
        text: Text to evaluate

    Returns:
        Coherence score (0-1)
    """
    if not text or len(text) < 10:
        return 0.0

    # Extract sentences
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if not sentences:
        return 0.0

    # Basic coherence: sentence length and count
    coherent_sentences = [s for s in sentences if len(s) > 15]  # Shorter threshold than original
    basic_coherence = len(coherent_sentences) / max(len(sentences), 1)

    # Check for discourse markers and semantic continuity
    continuity_markers = [
        "therefore", "thus", "consequently", "however", "moreover",
        "additionally", "furthermore", "because", "since", "then",
        "also", "in addition", "for example", "such as", "like",
        "unlike", "similar to", "but", "yet", "nevertheless", "instead",
        "while", "whereas", "despite", "although", "accordingly",
        "meanwhile", "subsequently", "meanwhile", "alternatively",
        "specifically", "notably", "importantly"
    ]

    # Count discourse markers (more suggests higher coherence)
    marker_count = sum(1 for marker in continuity_markers if marker in text.lower())
    marker_score = min(1.0, marker_count / 5)  # Cap at 5 markers

    # Check for coherent paragraph structure
    has_paragraphs = '\n\n' in text or '\n \n' in text
    paragraph_structure = 0.1 if has_paragraphs else 0.0

    # Check for topic consistency using simple subject repetition
    subjects = []
    for sentence in sentences:
        words = sentence.strip().split()
        if words and len(words) > 2:
            subjects.append(words[0].lower())  # Simple assumption that first word could be subject

    # Calculate topic coherence based on subject repetition
    if subjects:
        unique_subjects = len(set(subjects))
        subject_ratio = unique_subjects / len(subjects)

        # Lower ratio = more repeated subjects = more coherent topic
        topic_coherence = 1.0 - min(1.0, subject_ratio)
    else:
        topic_coherence = 0.5  # Default

    # Check for logical flow using sentence position words
    position_words = ["first", "second", "third", "finally", "lastly", "initially", "next", "then"]
    has_position_structure = any(word in text.lower() for word in position_words)
    position_score = 0.1 if has_position_structure else 0.0

    # Combine metrics with weights
    combined_coherence = (
            basic_coherence * 0.4 +
            topic_coherence * 0.25 +
            marker_score * 0.15 +
            paragraph_structure +
            position_score
    )

    return min(1.0, combined_coherence)


def assess_extended_criticality(metrics: Dict[str, float]) -> int:
    """
    Enhanced criticality assessment using multiple metrics with adjusted weights.

    This determines which zone the system is currently operating in:
    0 = Ordered zone (coherent but predictable)
    1 = Critical zone (sweet spot - the edge of chaos)
    2 = Chaotic zone (novel but incoherent)

    Args:
        metrics: Dictionary of criticality metrics

    Returns:
        Criticality zone (0, 1, or 2)
    """
    # Extract metrics
    token_likelihood = metrics.get("token_likelihood", 0.6)
    semantic_novelty = metrics.get("semantic_novelty", 0.5)
    coherence = metrics.get("coherence", 0.7)
    diversity = metrics.get("diversity", 0.6)
    surprise_factor = metrics.get("surprise_factor", 0.5)

    # Log all metrics for debugging
    print(f"[Debug] Token Likelihood: {token_likelihood:.3f}")
    print(f"[Debug] Semantic Novelty: {semantic_novelty:.3f}")
    print(f"[Debug] Coherence: {coherence:.3f}")
    print(f"[Debug] Token Diversity: {diversity:.3f}")
    print(f"[Debug] Surprise Factor: {surprise_factor:.3f}")

    # Calculate composite scores with adjusted weights
    coherence_weight = 0.4
    novelty_weight = 0.35
    diversity_weight = 0.25

    weighted_coherence = coherence * coherence_weight
    weighted_novelty = (semantic_novelty * 0.7 + surprise_factor * 0.3) * novelty_weight
    weighted_diversity = diversity * diversity_weight

    # Calculate combined creativity score (novelty + diversity)
    creative_score = weighted_novelty + weighted_diversity

    # Define adjusted thresholds for zone classification
    COHERENCE_THRESHOLD = 0.7  # Minimum coherence for critical or ordered zones
    NOVELTY_THRESHOLD_LOW = 0.3  # Minimum novelty for critical zone
    NOVELTY_THRESHOLD_HIGH = 0.6  # Maximum novelty before chaotic zone

    if (abs(weighted_coherence - COHERENCE_THRESHOLD) < 0.05 and
            abs(creative_score - ((NOVELTY_THRESHOLD_LOW + NOVELTY_THRESHOLD_HIGH) / 2)) < 0.1):
        print("[Near-boundary detected] Classifying as critical zone")
        return 1  # Close enough to critical zone boundaries

    # Calculate criticality score for meta-analysis
    criticality_score = weighted_coherence * (1.0 - abs(creative_score - 0.35) * 2)

    # Determine zone with modified rules
    if weighted_coherence > COHERENCE_THRESHOLD:
        if creative_score < NOVELTY_THRESHOLD_LOW:
            return 0  # Ordered zone: coherent but not creative enough
        elif creative_score > NOVELTY_THRESHOLD_HIGH:
            # Only go to chaotic if truly incoherent, otherwise keep in critical
            if coherence < 0.5:
                return 2  # Chaotic zone: high creativity, low coherence
            else:
                return 1  # Critical zone: coherent with high creativity
        else:
            return 1  # Critical zone: balanced novelty and coherence
    else:
        if creative_score > NOVELTY_THRESHOLD_LOW:
            return 2  # Chaotic zone: incoherent and novel
        else:
            return 0  # Ordered but still lacks coherence


def adaptive_interpolation(client, ordered_response: str, chaotic_response: str, query: str,
                           target_criticality: float = 0.8) -> str:
    """
    Creates a response in the edge of chaos by interpolating between an ordered and a chaotic response.

    This allows the system to blend the coherence of ordered responses with the
    creativity of chaotic responses to reach the sweet spot in between.

    Args:
        client: The OpenAI client
        ordered_response: A more structured, coherent response
        chaotic_response: A more creative, potentially less coherent response
        query: The original user query
        target_criticality: Target level of criticality (0-1)

    Returns:
        A blended response in the critical zone
    """
    # Assess component responses
    coherence_ordered = measure_internal_consistency(ordered_response)
    coherence_chaotic = measure_internal_consistency(chaotic_response)

    # Determine ideal blend weight based on coherence targets
    if coherence_ordered > 0.7 and coherence_chaotic > 0.5:
        # Both responses are high quality, use more balanced blending
        blend_weight = 0.5  # Even blend when both are good
        print("[Adaptive Interpolation] Both responses are high quality, using balanced blend")
    elif coherence_ordered < 0.6:
        # Ordered response isn't very coherent
        if coherence_chaotic > 0.6:
            # Chaotic response is actually more coherent
            blend_weight = 0.4  # Use more of the chaotic response
            print("[Adaptive Interpolation] Chaotic response more coherent, favoring it")
        else:
            # Neither is very coherent, slightly favor ordered
            blend_weight = 0.6
            print("[Adaptive Interpolation] Neither highly coherent, slightly favoring ordered")
    else:
        # Standard case - ordered is coherent, chaotic less so
        novelty_adjustment = min(0.2, (target_criticality - 0.5))  # Allow up to 0.2 adjustment
        blend_weight = 0.6 - novelty_adjustment  # Start with slight ordered bias, adjust toward target
        print(f"[Adaptive Interpolation] Standard case with novelty adjustment: {novelty_adjustment:.2f}")

        # Ensure blend_weight stays in reasonable range
    blend_weight = max(0.3, min(0.7, blend_weight))

    print(f"[Adaptive Interpolation] Blend weight: {blend_weight:.2f}")
    print(f"[Adaptive Interpolation] Coherence: ordered={coherence_ordered:.2f}, chaotic={coherence_chaotic:.2f}")

    # Generate interpolated response using LLM blending
    interpolation_prompt = f"""
    Combine these two AI responses into a single coherent response that has both structure and creativity.

    User Question: {query}

    More Structured Response: 
    {ordered_response}

    More Creative Response:
    {chaotic_response}

    The final response should be:
    1. Directly relevant to the user's question
    2. Balanced - approximately {int(blend_weight * 100)}% structured and {int((1 - blend_weight) * 100)}% creative


    Your response should read as a single, unified answer - not as a combination of two different responses.
    """

    try:
        # Use a lower temperature for more predictable blending
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": interpolation_prompt}],
            temperature=0.4,
            max_tokens=800
        )

        interpolated_response = response.choices[0].message.content

        # Basic validation - if too short, fall back to ordered response
        if len(interpolated_response.split()) < 20:
            print("[Adaptive Interpolation] Warning: Interpolated response too short, falling back")
            return ordered_response

        return interpolated_response
    except Exception as e:
        print(f"[Adaptive Interpolation] Error: {e}")
        # Fall back to ordered response in case of error
        return ordered_response