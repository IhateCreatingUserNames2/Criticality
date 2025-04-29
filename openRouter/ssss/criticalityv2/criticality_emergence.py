"""
Criticality Emergence System

This module implements an emergent approach to finding the critical zone between
order and chaos using principles from quantum stochastic field theory.

Instead of hard-coded parameter adjustments, it creates a dynamic phase space
where criticality can emerge through attractor dynamics and self-organization.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import time


class CriticalityEmergenceSystem:
    """
    A system that allows criticality to emerge through dynamic interactions
    in a parameter phase space, inspired by quantum stochastic field theory
    and edge of chaos principles.
    """

    def __init__(self, memory_blossom, criticality_memory):
        self.memory_blossom = memory_blossom
        self.criticality_memory = criticality_memory

        # Phase space representation
        self.parameter_landscape = {}
        self.attractor_points = []
        self.repeller_points = []

        # Quantum stochastic field parameters
        self.field_temperature = 0.7
        self.field_coupling = 0.3
        self.noise_amplitude = 0.1

        # Emergence tracking
        self.transition_history = []
        self.entropy_states = []

        # Initialize parameter landscape from memory
        self._initialize_parameter_landscape()

    def _initialize_parameter_landscape(self):
        """Initialize the parameter landscape from criticality memory."""
        # Extract successful parameter sets from memory
        if hasattr(self.criticality_memory, 'phase_space'):
            critical_params = self.criticality_memory.phase_space.get('critical_region', [])
            ordered_params = self.criticality_memory.phase_space.get('ordered_region', [])
            chaotic_params = self.criticality_memory.phase_space.get('chaotic_region', [])

            # Initialize attractor points in parameter space
            if critical_params:
                for p in critical_params[:5]:
                    # Convert dict to tuple of values for temperature, top_p, freq_penalty, pres_penalty
                    param_tuple = (
                        p.get('temperature', 0.7),
                        p.get('top_p', 0.9),
                        p.get('frequency_penalty', 0.3),
                        p.get('presence_penalty', 0.3)
                    )
                    self.attractor_points.append(param_tuple)

            # Initialize repeller points
            for p in ordered_params[:3]:
                if p.get('temperature', 0.7) < 0.3 or p.get('top_p', 0.9) < 0.6:
                    param_tuple = (
                        p.get('temperature', 0.7),
                        p.get('top_p', 0.9),
                        p.get('frequency_penalty', 0.3),
                        p.get('presence_penalty', 0.3)
                    )
                    self.repeller_points.append(param_tuple)

            for p in chaotic_params[:3]:
                if p.get('temperature', 0.7) > 1.3 or p.get('top_p', 0.9) > 0.98:
                    param_tuple = (
                        p.get('temperature', 0.7),
                        p.get('top_p', 0.9),
                        p.get('frequency_penalty', 0.3),
                        p.get('presence_penalty', 0.3)
                    )
                    self.repeller_points.append(param_tuple)

    def generate_emergent_response(self, client, messages, query, query_analysis,
                                   calculate_metrics_func, assess_criticality_func,
                                   model_name="gpt-4o-mini"):
        """
        Generate a response that emerges at the edge of chaos through
        dynamic interaction of parameter fields.

        Args:
            client: OpenAI client
            messages: Message list
            query: User query
            query_analysis: Query analysis
            calculate_metrics_func: Function to calculate metrics
            assess_criticality_func: Function to assess criticality zone

        Returns:
            Tuple of (response, zone, metrics)
        """
        # Start with either memory-based parameters or exploration parameters
        params = self._get_starting_parameters(query, query_analysis)

        # Create a field-theoretic trajectory through parameter space
        parameter_trajectory = self._generate_parameter_trajectory(params, query_analysis)

        # Track responses and their characteristics
        responses = []
        criticalness_measures = []

        # Try up to 3 points along the trajectory (emergent sampling)
        for i, trajectory_params in enumerate(parameter_trajectory[:3]):
            try:
                # Convert parameters to API format
                api_params = {
                    "temperature": trajectory_params[0],
                    "top_p": trajectory_params[1],
                    "frequency_penalty": trajectory_params[2],
                    "presence_penalty": trajectory_params[3],
                    "max_tokens": 800,
                    "logprobs": True,
                    "top_logprobs": 5
                }

                print(f"[Emergent System] Trying parameter point {i + 1}: temp={api_params['temperature']:.2f}, "
                      f"top_p={api_params['top_p']:.2f}")

                # Generate response
                # Generate response
                response = client.chat.completions.create(
                    model=model_name,  # Use the passed model name
                    messages=messages,
                    **api_params
                )
                response_text = response.choices[0].message.content

                # Measure criticality using provided functions
                metrics = calculate_metrics_func(response_text, response, query)
                zone = assess_criticality_func(metrics)

                print(f"[Emergent System] Point {i + 1} result: Zone {zone}, "
                      f"Coherence: {metrics.get('coherence', 0):.2f}, "
                      f"Novelty: {metrics.get('semantic_novelty', 0):.2f}")

                # Store response data
                responses.append((response_text, zone, metrics))
                criticalness_measures.append(self._calculate_criticalness(metrics))

                # If we've reached critical zone, no need for more samples
                if zone == 1:
                    # Update parameter landscape with this successful point
                    self._update_parameter_landscape(trajectory_params, 1.0)
                    return response_text, zone, metrics

            except Exception as e:
                print(f"[Parameter trajectory point {i + 1} failed] {e}")
                continue

        # If we have responses but none in critical zone, use the closest one
        if responses:
            # Find response closest to critical zone
            best_idx = np.argmax(criticalness_measures)
            best_response, best_zone, best_metrics = responses[best_idx]

            print(f"[Emergent System] No critical zone reached. Using best response "
                  f"(criticalness: {criticalness_measures[best_idx]:.2f}) from point {best_idx + 1}")

            # Update parameter landscape with feedback
            self._update_parameter_landscape(parameter_trajectory[best_idx],
                                             criticalness_measures[best_idx])

            return best_response, best_zone, best_metrics

        # Fallback if all attempts failed
        print("[Emergent System] All trajectory points failed. Using fallback parameters.")
        fallback_params = {
            "temperature": 0.7,
            "max_tokens": 800
        }

        try:
            # Fallback if all attempts failed
            fallback = client.chat.completions.create(
                model=model_name,  # Use the passed model name
                messages=messages,
                **fallback_params
            )
            fallback_text = fallback.choices[0].message.content
            fallback_metrics = calculate_metrics_func(fallback_text, fallback, query)
            fallback_zone = assess_criticality_func(fallback_metrics)

            return fallback_text, fallback_zone, fallback_metrics
        except Exception as e:
            print(f"[Fallback failed] {e}")
            return "I apologize, but I'm having trouble generating a response right now.", 1, {}

    def _get_starting_parameters(self, query, query_analysis):
        """Get starting parameters from memory or exploration."""
        # Check memory for similar queries
        memory_params = None
        if hasattr(self.criticality_memory, 'get_closest_params'):
            memory_params = self.criticality_memory.get_closest_params(query)

        if memory_params:
            # Extract parameter values as a tuple
            return (
                memory_params.get("temperature", 0.7),
                memory_params.get("top_p", 0.9),
                memory_params.get("frequency_penalty", 0.3),
                memory_params.get("presence_penalty", 0.3)
            )

        # If no memory matches, use query analysis for initial parameters
        complexity = query_analysis.get("complexity", 0.5)
        sentiment = query_analysis.get("sentiment", 0.0)

        # Complexity affects temperature and top_p
        temp_base = 0.7 - (complexity - 0.5) * 0.2
        top_p_base = 0.85 + (complexity - 0.5) * 0.1

        # Sentiment affects penalties
        freq_penalty = 0.3 + abs(sentiment) * 0.2
        pres_penalty = 0.3 + abs(sentiment) * 0.2

        return (temp_base, top_p_base, freq_penalty, pres_penalty)

    def _generate_parameter_trajectory(self, start_params, query_analysis):
        """
        Generate a trajectory through parameter space using
        principles from quantum stochastic field theory.
        """
        # Initialize trajectory with starting point
        trajectory = [start_params]

        # Number of steps in trajectory
        steps = 5

        # Current position
        current = np.array(start_params)

        for _ in range(steps - 1):
            # Calculate field forces from attractors and repellers
            attractor_force = np.zeros(4)
            if self.attractor_points:
                for attractor in self.attractor_points:
                    attractor_array = np.array(attractor)
                    direction = attractor_array - current
                    distance = np.linalg.norm(direction) + 1e-6  # Avoid division by zero
                    attractor_force += direction / (distance ** 2) * self.field_coupling

            repeller_force = np.zeros(4)
            if self.repeller_points:
                for repeller in self.repeller_points:
                    repeller_array = np.array(repeller)
                    direction = current - repeller_array
                    distance = np.linalg.norm(direction) + 1e-6
                    repeller_force += direction / (distance ** 2) * self.field_coupling

            # Add stochastic noise (quantum fluctuations)
            noise = np.random.normal(0, self.noise_amplitude, 4)

            # Combine forces with field temperature (higher = more random movement)
            movement = (attractor_force + repeller_force) * (
                        1 - self.field_temperature) + noise * self.field_temperature

            # Move to new position
            current += movement

            # Constrain parameters to reasonable ranges
            current[0] = np.clip(current[0], 0.2, 1.5)  # temperature
            current[1] = np.clip(current[1], 0.5, 1.0)  # top_p
            current[2] = np.clip(current[2], 0.0, 1.0)  # frequency_penalty
            current[3] = np.clip(current[3], 0.0, 1.0)  # presence_penalty

            # Add to trajectory
            trajectory.append(tuple(current))

        return trajectory

    def _update_parameter_landscape(self, params, performance):
        """
        Update the parameter landscape based on the performance
        of a particular parameter set.
        """
        # If high performance, add as an attractor point
        if performance > 0.8:
            self.attractor_points.append(params)
            # Keep only the most recent points
            if len(self.attractor_points) > 10:
                self.attractor_points.pop(0)

        # If very low performance, add as a repeller point
        elif performance < 0.3:
            self.repeller_points.append(params)
            if len(self.repeller_points) > 10:
                self.repeller_points.pop(0)

        # Update field temperature based on recent performance
        # Higher performance = lower temperature = more exploitation
        # Lower performance = higher temperature = more exploration
        self.field_temperature = max(0.3, min(0.9, 1.0 - performance))

        print(f"[Emergent System] Field temperature updated to {self.field_temperature:.2f}")
        print(f"[Emergent System] Attractors: {len(self.attractor_points)}, Repellers: {len(self.repeller_points)}")

    def _calculate_criticalness(self, metrics):
        """
        Calculate how close a response is to the critical zone
        based on its metrics.
        """
        # Extract key metrics
        coherence = metrics.get("coherence", 0.5)
        novelty = metrics.get("semantic_novelty", 0.5)
        diversity = metrics.get("diversity", 0.5)

        # Ideal values for critical zone
        ideal_coherence = 0.75
        ideal_novelty = 0.5
        ideal_diversity = 0.6

        # Calculate distance from ideal (lower is better)
        coherence_distance = abs(coherence - ideal_coherence)
        novelty_distance = abs(novelty - ideal_novelty)
        diversity_distance = abs(diversity - ideal_diversity)

        # Weighted distance (coherence most important)
        weighted_distance = (
                coherence_distance * 0.5 +
                novelty_distance * 0.3 +
                diversity_distance * 0.2
        )

        # Convert to score (higher is better)
        return max(0.0, 1.0 - weighted_distance * 2)