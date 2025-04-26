# Memory Blossom

An advanced cognitive architecture for AI assistants that combines multi-layered memory systems with adaptive criticality assessment.

## Overview

Memory Blossom is a sophisticated system designed to enhance AI assistants with human-like memory organization and optimal response generation. It works by:

1. Storing memories in specialized categories using neurally-inspired embedding models
2. Dynamically retrieving contextually relevant memories during conversations
3. Assessing response quality across order-chaos dimensions
4. Adaptively tuning responses to find an optimal "sweet spot" between predictability and novelty

## Theory: Why Memory Blossom Matters

This architecture addresses two key limitations in current AI systems:

### 1. The Memory Problem

Standard large language models lack true episodic memory. They operate primarily on their training data and immediate conversation context, without building persistent, contextually-aware memories over time.

Memory Blossom implements a multi-layered memory system inspired by human cognitive psychology:

- **Explicit Memory**: Factual knowledge and information
- **Emotional Memory**: Affective experiences and feelings
- **Procedural Memory**: Skills and know-how
- **Flashbulb Memory**: Vivid, identity-defining experiences
- **Somatic Memory**: Sensory and physical experiences
- **Liminal Memory**: Emerging ideas and possibilities
- **Generative Memory**: Creative and imaginative concepts

Each memory type uses a specialized neural embedding model to optimize semantic representation specific to that category of information.

### 2. The Criticality Problem

AI responses often fall into one of two suboptimal zones:

- **The Ordered Zone**: Responses that are coherent but predictable/generic
- **The Chaotic Zone**: Responses that are novel but potentially incoherent

Memory Blossom implements a "criticality assessment" system that evaluates responses along three dimensions:

1. **Statistical Likelihood**: How predictable is the response based on language model probabilities?
2. **Semantic Novelty**: How different is the response from the input query?
3. **Internal Consistency**: How coherent is the response's internal structure?

This assessment helps identify the "Critical Zone" - where responses are both coherent and novel - and adjusts generation parameters to reach this optimal state.

## Components

### memory_blossom.py

This file implements the core memory system:

- `Memory` class: Stores individual memories with rich metadata
- `MemoryBlossom` class: Manages memory categorization and retrieval
- Specialized embedding models: Seven different neural embedding models for specific memory types
- Dynamic classification: Uses GPT-3.5-Turbo to classify memories (with fallback heuristics)
- Context-aware retrieval: Analyzes conversation flow to prioritize relevant memory types
- Persistence: Save/load functionality for long-term memory retention

### chat.py

This file implements the conversational agent:

- `MemoryBlossomChatbot` class: Integrates Memory Blossom with OpenAI's API
- Criticality assessment: Evaluates response quality along order-chaos spectrum
- Adaptive regeneration: Adjusts temperature parameter to reach optimal response zone
- Command-line interface: Simple REPL for interactive testing
- System prompt injection: Incorporates relevant memories into prompts

## Usage

1. Install dependencies:
```bash
pip install sentence-transformers sklearn numpy openai python-dotenv
```

2. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key"
```

3. Run the interactive chat:
```bash
python chat.py
```

## Command Line Interface

The simple CLI supports:
- Normal chat interaction
- `memories` command to view recent memories
- `clear` command to reset conversation context
- `exit` command to quit

## Extensions and Customization

- Add custom memory types by extending the memory_stores dictionary
- Implement more sophisticated emotion scoring by replacing the keyword-based approach
- Use different LLMs by changing the OpenAI model parameter
- Adjust criticality thresholds to tune the sweet spot between novelty and coherence

## References & Inspiration

This system draws inspiration from:
- Cognitive psychology models of human memory
- Complex systems theory, particularly self-organized criticality
- Neural embedding techniques for semantic representation
- LLM prompt engineering best practices

## Requirements

- Python 3.8+
- Recommended: 32GB+ RAM (for multiple embedding models)
- Optional: NVIDIA GPU with 4GB+ VRAM
- Internet connection for OpenAI API calls
