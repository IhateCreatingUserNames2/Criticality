"""
Edge of Chaos AI System

This main module implements the complete Edge of Chaos AI system, integrating:
1. Memory Blossom - Advanced multi-modal memory system
2. Memory Connector - Dynamic relationship discovery between memories
3. Criticality Control - Parameter management for edge of chaos operation
4. Narrative Context Framing - Enhanced knowledge integration techniques

Together, these systems create an AI that operates at the critical boundary between
order and chaos, balancing coherence with creativity for optimal performance.
"""

import os
import argparse
from dotenv import load_dotenv
from enhanced_chat import EnhancedMemoryBlossomChat
from memory_blossom import save_memories, save_chat_history

# Load environment variables
load_dotenv()


def get_api_details():
    """Get API details from environment or user input."""
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

    if not api_key:
        print("OpenAI API key not found in environment variables.")
        api_key = input("Please enter your OpenAI API key: ")

    return api_key, api_base


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Edge of Chaos AI System")

    parser.add_argument('--model', type=str, default="gpt-4o-mini",
                        help="Model to use (default: gpt-4o-mini)")
    parser.add_argument('--openrouter', action='store_true',
                        help="Use OpenRouter instead of OpenAI")
    parser.add_argument('--no-persistence', dest='persistence', action='store_false',
                        help="Disable memory persistence")
    parser.add_argument('--debug', action='store_true',
                        help="Enable debug output")

    return parser.parse_args()


def setup_openrouter():
    """Setup OpenRouter API details."""
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    if not openrouter_key:
        print("OpenRouter API key not found in environment variables.")
        openrouter_key = input("Please enter your OpenRouter API key: ")

    return openrouter_key, "https://openrouter.ai/api/v1"


def print_welcome_message():
    """Print welcome message with system information."""
    welcome_text = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   Edge of Chaos AI System                                 ║
    ║   --------------------------------------------------     ║
    ║                                                           ║
    ║   Operating at the critical boundary between              ║
    ║   order and chaos for optimal information processing      ║
    ║                                                           ║
    ║   Key Components:                                         ║
    ║   * Memory Blossom - Multi-modal memory system            ║
    ║   * Memory Connector - Relationship discovery             ║
    ║   * Criticality Control - Parameter management            ║
    ║   * Narrative Context Framing - Knowledge integration     ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝

    Type 'memories' to view recent memories
    Type 'clear' to wipe conversation context
    Type 'state' to view current system state
    Type 'help' to see these commands again
    Type 'exit' to quit
    """
    print(welcome_text)


def print_help():
    """Print help information."""
    help_text = """
    Available Commands:
    ------------------
    memories    - View recent memories
    clear       - Clear conversation history
    state       - View current system state
    add <type>  - Add a new memory (types: explicit, emotional, procedural, 
                  flashbulb, somatic, liminal, generative)
    debug       - Toggle debug output
    help        - Show this help message
    exit        - Exit the program
    """
    print(help_text)


def handle_add_memory(bot, command_parts):
    """Handle adding a new memory."""
    if len(command_parts) < 2:
        print("Please specify a memory type: explicit, emotional, procedural, flashbulb, somatic, liminal, generative")
        return

    memory_type = command_parts[1].capitalize()
    valid_types = ["Explicit", "Emotional", "Procedural", "Flashbulb", "Somatic", "Liminal", "Generative"]

    if memory_type not in valid_types:
        print(f"Invalid memory type. Please use one of: {', '.join(valid_types).lower()}")
        return

    content = input("Enter memory content: ")
    emotion_score = float(input("Enter emotion score (0.0-1.0): "))

    bot.add_memory(content=content, memory_type=memory_type, emotion_score=emotion_score)
    print(f"Memory added with type '{memory_type}'")


def main():
    """Main function to run the Edge of Chaos AI system."""
    args = parse_arguments()

    # Get API details
    if args.openrouter:
        api_key, api_base = setup_openrouter()
    else:
        api_key, api_base = get_api_details()

    # Print welcome message
    print_welcome_message()

    # Initialize the enhanced chat system
    print("Initializing Edge of Chaos AI system...")
    bot = EnhancedMemoryBlossomChat(
        api_key=api_key,
        api_base=api_base,
        model=args.model,
        memory_persistence=args.persistence
    )
    print("Initialization complete.")

    # Debug mode flag
    debug_mode = args.debug

    # Main conversation loop
    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            # Check for special commands
            if user_input.lower() == "exit":
                print("Goodbye!")
                break

            elif user_input.lower() == "memories":
                mems = bot.retrieve_relevant_memories("recent", top_k=5)
                print("\n--- Recent Memories ---")
                for i, mem in enumerate(mems, 1):
                    print(f"{i}. [{mem['type']}] {mem['content'][:100]}{'...' if len(mem['content']) > 100 else ''}")
                continue

            elif user_input.lower() == "clear":
                bot.clear_conversation_history()
                print("Conversation history cleared.")
                continue

            elif user_input.lower() == "state":
                print("\n--- Current System State ---")
                print(f"Criticality Zone: {bot.current_state['criticality_zone']}")
                print(f"Recent Topics: {', '.join(bot.current_state['recent_topics'])}")
                print(f"Active Frames: {len(bot.current_state['active_frames'])}")
                print(f"Temperature: {bot.temperature:.2f}")
                continue

            elif user_input.lower() == "save":
                print("Saving system state...")
                memories_saved = save_memories(bot.memory_blossom)
                history_saved = save_chat_history(bot.conversation_history)
                if memories_saved and history_saved:
                    print("System state successfully saved.")
                else:
                    print("There were issues saving the system state. Check the logs for details.")
                continue

            elif user_input.lower() == "help":
                print_help()
                continue

            elif user_input.lower() == "debug":
                debug_mode = not debug_mode
                print(f"Debug mode {'enabled' if debug_mode else 'disabled'}")
                continue

            elif user_input.lower().startswith("add "):
                command_parts = user_input.split()
                handle_add_memory(bot, command_parts)
                continue

            # Normal conversation - get AI response
            if debug_mode:
                print("\n[Debug] Analyzing query and retrieving memories...")

            ai_response = bot.chat(user_input)

            if debug_mode:
                print(f"\n[Debug] Response generated in zone {bot.current_state['criticality_zone']}")
                print(f"[Debug] Coherence: {bot.metric_history.get('coherence', [0])[-1]:.2f}, "
                      f"Novelty: {bot.metric_history.get('semantic_novelty', [0])[-1]:.2f}")

            print("\nAI:", ai_response)

        except KeyboardInterrupt:
            print("\nInterrupted. Type 'exit' to quit.")
        except Exception as e:
            print(f"\nError: {e}")
            if debug_mode:
                import traceback
                traceback.print_exc()
            else:
                print("Run with --debug for more details")


if __name__ == "__main__":
    main()