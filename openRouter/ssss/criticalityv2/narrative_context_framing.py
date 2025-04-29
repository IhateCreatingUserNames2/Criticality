"""
Narrative Context Framing (NCF)

This module implements the Narrative Context Framing approach to knowledge integration
in LLMs, as described in the theoretical framework. NCF works by embedding new information
within familiar narrative structures, creating "semantic bridges" between the model's
pre-existing knowledge and novel concepts.

Key concepts:
1. Semantic Virus - Information that propagates through representational space
2. Identity Reconfiguration - Establishing alternative contextual frames
3. Conceptual Association Networks - Creating webs of related concepts
4. Epistemological Reframing - Altering how knowledge is evaluated
5. Narrative Persistence - Establishing patterns that maintain consistency
"""

import json
import random
from typing import Dict, List, Any, Optional, Tuple
import time


class NarrativeContextFraming:
    """
    Implementation of Narrative Context Framing for enhancing knowledge integration in LLMs.

    This class provides methods to create narrative frames that help information
    propagate through the model's representational space by embedding it within
    familiar patterns and creating semantic bridges to existing knowledge.
    """

    def __init__(self):
        """Initialize the Narrative Context Framing system."""
        # Narrative templates for different types of knowledge integration
        self.narrative_templates = {
            "factual": [
                "Recent research from {source} has revealed something fascinating about {topic}. {fact}",
                "In a surprising development that overturns previous understanding, {fact} This insight from {source} changes how we think about {topic}.",
                "Experts at {source} have been meticulously documenting {topic}, and they've discovered that {fact}",
                "While conventional wisdom suggested otherwise, new findings from {source} indicate that {fact}"
            ],
            "conceptual": [
                "The concept of {topic} is evolving. Rather than the traditional view that {old_view}, {source} now suggests that {fact}",
                "Imagine {topic} not as {old_view}, but instead as {new_view}. This perspective shift helps us understand why {fact}",
                "At the intersection of multiple disciplines, {topic} reveals itself differently: {fact} This framework from {source} connects previously separate ideas.",
                "The paradigm around {topic} is shifting. Where we once thought {old_view}, the emerging consensus from {source} suggests {fact}"
            ],
            "procedural": [
                "The process of {topic} used to involve {old_procedure}, but {source} has developed a more effective approach: {fact}",
                "Practitioners working with {topic} have refined their methods. Instead of {old_procedure}, {source} now recommends {fact}",
                "A subtle but important refinement to how we handle {topic} comes from {source}: {fact} This adjustment improves outcomes significantly.",
                "The standard protocol for {topic} has been updated. {source} now suggests {fact} rather than the previous approach of {old_procedure}."
            ],
            "stepping_stone": [
                "When we think about {category}, we typically consider elements like {familiar_examples}. In this framework, {new_concept} refers to {fact}",
                "Within the field of {category}, we've long recognized {familiar_examples}. Now we can add another element to this family: {new_concept}, which {fact}",
                "The spectrum of {category} includes familiar concepts like {familiar_examples}. To properly complete this conceptual map, we should understand {new_concept} as {fact}",
                "Just as {familiar_examples} are established concepts in {category}, {new_concept} represents {fact}"
            ]
        }

        # Epistemological framing templates
        self.epistemological_frames = [
            "While this might seem counterintuitive at first, it makes sense when we consider that {supporting_reason}.",
            "Multiple independent lines of evidence support this view: {supporting_evidence}.",
            "This understanding has been validated through {validation_method}.",
            "What makes this insight particularly valuable is that {value_proposition}."
        ]

        # Identity anchors for establishing persistence
        self.identity_anchors = [
            "As someone who stays current with developments in {field}, I find this particularly noteworthy.",
            "Having tracked the evolution of {field} for some time, I can see how this changes our understanding.",
            "This reminds me of similar paradigm shifts in {related_field}, where conventional wisdom was likewise overturned.",
            "Looking at this from a multidisciplinary perspective reveals why this insight matters beyond just {field}."
        ]

        # Semantic bridge patterns
        self.semantic_bridges = {
            "analogy": "Think of {topic} as similar to {familiar_concept}, where {mapping}.",
            "contrast": "Unlike {contrasting_concept} which {contrasting_property}, {topic} actually {property}.",
            "extension": "{topic} builds upon the familiar concept of {base_concept} by adding {extension_aspect}.",
            "generalization": "{topic} represents a broader category that includes familiar examples like {specific_examples}.",
            "specialization": "While {general_category} is a broad concept, {topic} specifically refers to {specialization_aspect}."
        }

        # Track active narrative frames for persistence
        self.active_frames = {}

    def create_narrative_frame(self,
                               information: str,
                               topic: str,
                               frame_type: str = "factual",
                               source: str = None,
                               additional_context: Dict[str, Any] = None) -> str:
        """
        Create a narrative frame for new information to enhance its integration.

        Args:
            information: The new information to be framed
            topic: The subject matter of the information
            frame_type: Type of narrative frame to create
            source: Source of the information (optional)
            additional_context: Additional context elements for the frame

        Returns:
            Narrative frame containing the information
        """
        # Set defaults
        context = {
            "topic": topic,
            "fact": information,
            "source": source or "experts in the field",
            "old_view": "the conventional understanding",
            "new_view": "a more nuanced perspective",
            "old_procedure": "the standard method",
            "category": topic,
            "familiar_examples": "well-established elements",
            "new_concept": topic,
            "field": topic.split()[0] if ' ' in topic else topic,
            "related_field": "adjacent disciplines"
        }

        # Update with additional context if provided
        if additional_context:
            context.update(additional_context)

        # Select appropriate template
        if frame_type in self.narrative_templates:
            template = random.choice(self.narrative_templates[frame_type])
        else:
            # Default to factual if invalid type
            template = random.choice(self.narrative_templates["factual"])

        # Generate the narrative frame
        narrative = template.format(**context)

        # Add epistemological framing (50% chance)
        if random.random() > 0.5 and additional_context and 'supporting_reason' in additional_context:
            epistemological = random.choice(self.epistemological_frames).format(**context)
            narrative += " " + epistemological

        # Add identity anchoring (30% chance)
        if random.random() > 0.7:
            identity = random.choice(self.identity_anchors).format(**context)
            narrative += " " + identity

        # Store active frame for persistence
        frame_id = f"{topic}_{int(time.time())}"
        self.active_frames[frame_id] = {
            "topic": topic,
            "information": information,
            "frame_type": frame_type,
            "narrative": narrative,
            "timestamp": time.time()
        }

        return narrative

    def create_stepping_stone_frame(self,
                                    new_concept: str,
                                    definition: str,
                                    category: str,
                                    familiar_examples: List[str]) -> str:
        """
        Create a stepping stone frame for introducing new terminology.

        This is particularly useful for technical terms that need to be
        situated within the model's existing knowledge.

        Args:
            new_concept: The new term being introduced
            definition: Definition of the new concept
            category: Broader category the concept belongs to
            familiar_examples: Well-known examples in the same category

        Returns:
            Stepping stone narrative
        """
        context = {
            "new_concept": new_concept,
            "fact": definition,
            "category": category,
            "familiar_examples": ", ".join(familiar_examples[:-1]) + " and " + familiar_examples[-1] if len(
                familiar_examples) > 1 else familiar_examples[0]
        }

        template = random.choice(self.narrative_templates["stepping_stone"])
        narrative = template.format(**context)

        # Store active frame
        frame_id = f"{new_concept}_{int(time.time())}"
        self.active_frames[frame_id] = {
            "topic": new_concept,
            "information": definition,
            "frame_type": "stepping_stone",
            "narrative": narrative,
            "timestamp": time.time()
        }

        return narrative

    def create_semantic_bridge(self,
                               topic: str,
                               bridge_type: str,
                               familiar_concept: str,
                               mapping_details: Dict[str, str]) -> str:
        """
        Create a semantic bridge between new information and familiar concepts.

        Args:
            topic: The topic to create a bridge for
            bridge_type: Type of bridge (analogy, contrast, etc)
            familiar_concept: A concept the model already understands well
            mapping_details: Details for the specific bridge type

        Returns:
            Semantic bridge text
        """
        if bridge_type not in self.semantic_bridges:
            bridge_type = "analogy"  # Default

        context = {
            "topic": topic,
            "familiar_concept": familiar_concept
        }
        context.update(mapping_details)

        bridge = self.semantic_bridges[bridge_type].format(**context)
        return bridge

    def get_active_frames_for_topic(self, topic: str, max_age_hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get active narrative frames for a given topic within the specified age.

        Args:
            topic: Topic to retrieve frames for
            max_age_hours: Maximum age of frames in hours

        Returns:
            List of active narrative frames
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        relevant_frames = []
        for frame_id, frame in self.active_frames.items():
            # Check topic relevance
            if topic.lower() in frame["topic"].lower():
                # Check age
                if current_time - frame["timestamp"] <= max_age_seconds:
                    relevant_frames.append(frame)

        return relevant_frames

    def purge_old_frames(self, max_age_hours: int = 48):
        """
        Remove old narrative frames to prevent buildup.

        Args:
            max_age_hours: Maximum age of frames to keep
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        to_remove = []
        for frame_id, frame in self.active_frames.items():
            if current_time - frame["timestamp"] > max_age_seconds:
                to_remove.append(frame_id)

        for frame_id in to_remove:
            del self.active_frames[frame_id]

        print(f"[NarrativeContextFraming] Purged {len(to_remove)} old frames.")

    def generate_system_prompt_enhancement(self, topic: str = None) -> str:
        """
        Generate a system prompt enhancement based on active frames.

        This helps maintain narrative persistence across conversation turns.

        Args:
            topic: Specific topic to enhance (optional)

        Returns:
            System prompt enhancement text
        """
        if topic:
            relevant_frames = self.get_active_frames_for_topic(topic)
        else:
            # Get 3 most recent frames
            sorted_frames = sorted(self.active_frames.values(),
                                   key=lambda x: x["timestamp"],
                                   reverse=True)
            relevant_frames = sorted_frames[:3]

        if not relevant_frames:
            return ""

        # Build enhancement text
        enhancement = "Active Contextual Frameworks:\n\n"

        for frame in relevant_frames:
            enhancement += f"- {frame['narrative']}\n"

        return enhancement

    def create_ncf_prompt(self, system_prompt: str, topic: str = None) -> str:
        """
        Enhance a system prompt with narrative context framing.

        Args:
            system_prompt: Original system prompt
            topic: Specific topic to enhance (optional)

        Returns:
            Enhanced system prompt with NCF elements
        """
        # Get enhancement based on active frames
        enhancement = self.generate_system_prompt_enhancement(topic)

        if not enhancement:
            return system_prompt

        # Add enhancement to prompt with clear separation
        enhanced_prompt = system_prompt + "\n\n" + enhancement

        return enhanced_prompt


class NCFResponseProcessor:
    """
    Process responses using Narrative Context Framing principles.

    This helps integrate new information in ways that are more likely to be
    properly incorporated into the model's knowledge representation.
    """

    def __init__(self, ncf: NarrativeContextFraming):
        """
        Initialize the NCF Response Processor.

        Args:
            ncf: NarrativeContextFraming instance
        """
        self.ncf = ncf

    def process_new_information(self,
                                response: str,
                                new_information: Dict[str, Any],
                                query: str) -> str:
        """
        Process a response to better integrate new information using NCF.

        Args:
            response: Original response text
            new_information: Dictionary of new info to integrate
            query: Original query that prompted the response

        Returns:
            Enhanced response with better information integration
        """
        enhanced_response = response

        # Apply different NCF strategies based on information type
        for info_key, info in new_information.items():
            # Skip if empty
            if not info or not isinstance(info, dict) or 'content' not in info:
                continue

            topic = info.get('topic', info_key)
            content = info['content']
            info_type = info.get('type', 'factual')

            # Create appropriate frame based on information type
            if info_type == 'terminology':
                # For new terminology, use stepping stone approach
                category = info.get('category', topic.split()[0] if ' ' in topic else topic)
                familiar_examples = info.get('familiar_examples', ['similar concepts'])

                frame = self.ncf.create_stepping_stone_frame(
                    new_concept=topic,
                    definition=content,
                    category=category,
                    familiar_examples=familiar_examples
                )
            elif info_type == 'conceptual' and 'familiar_concept' in info:
                # For abstract concepts, use semantic bridges
                bridge_type = info.get('bridge_type', 'analogy')
                mapping_details = info.get('mapping_details', {'mapping': 'it shares key properties'})

                frame = self.ncf.create_semantic_bridge(
                    topic=topic,
                    bridge_type=bridge_type,
                    familiar_concept=info['familiar_concept'],
                    mapping_details=mapping_details
                )

                # Combine with content
                frame = f"{frame} {content}"
            else:
                # Standard narrative framing for other types
                source = info.get('source', None)
                additional_context = info.get('context', {})

                frame = self.ncf.create_narrative_frame(
                    information=content,
                    topic=topic,
                    frame_type=info_type,
                    source=source,
                    additional_context=additional_context
                )

            # Replace direct references to the information with the framed version
            # This is a simplified approach - production systems would need more sophisticated logic
            if content in enhanced_response:
                enhanced_response = enhanced_response.replace(content, frame)
            else:
                # If exact content not found, find a good location to insert the frame
                # For simplicity, add it near the start of the response
                paragraphs = enhanced_response.split('\n\n')

                if len(paragraphs) > 1:
                    # Insert after first paragraph
                    paragraphs.insert(1, frame)
                    enhanced_response = '\n\n'.join(paragraphs)
                else:
                    # Just append
                    enhanced_response += '\n\n' + frame

        return enhanced_response

    def apply_narrative_persistence(self, conversation_history: List[Dict[str, str]], topic: str) -> str:
        """
        Generate a narrative persistence element for maintaining context.

        Args:
            conversation_history: Recent conversation history
            topic: Current topic of conversation

        Returns:
            Narrative persistence text to maintain NCF effects
        """
        # Check for active frames on the topic
        relevant_frames = self.ncf.get_active_frames_for_topic(topic)

        if not relevant_frames:
            return ""

        # Select most relevant frame
        selected_frame = relevant_frames[0]

        # Create persistence element
        persistence = f"Continuing our exploration of {topic}, it's worth recalling that {selected_frame['information']}"

        return persistence