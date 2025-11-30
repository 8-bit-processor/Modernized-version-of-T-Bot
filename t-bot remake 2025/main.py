import argparse
import logging
import hashlib
from typing import List, Tuple
from ollama_module import parse_text_to_objects, generate_text_embedding
from object_parser import parse_json_to_tbot_objects
from persistence_manager import add_analysis_to_knowledge_base, load_knowledge_base, find_analysis_by_text, find_best_cached_analysis, Analysis
from mathematical_reasoning_module import perform_mathematical_reasoning, combine_truths_across_sentences
from t_bot_objects import TBotObject

# Define a default file for storing parsed objects and embeddings
STORAGE_FILE = "tbot_analysis_data.json"

def analyze_single_sentence(user_input: str, knowledge_base: List[Analysis], similarity_threshold: float = 0.85) -> None:
    """Analyze a single sentence using mathematical reasoning."""
    try:
        logging.info("\nAnalyzing text (with cache check)...")

        # 1) exact-text cache
        cached = find_analysis_by_text(user_input, STORAGE_FILE)
        if cached:
            logging.info("Cache hit: reusing exact-text analysis")
            text_embedding = cached["original_text_embedding"]
            parsed_objects = cached["parsed_objects"]
        else:
            logging.info("Cache miss: computing embedding for semantic fallback")
            # compute embedding (needed for semantic lookup)
            text_embedding = generate_text_embedding(user_input)
            if not text_embedding:
                logging.warning("Failed to generate embedding.")
                text_embedding = []

            # 2) semantic fallback: try to find a similar cached analysis
            semantic_cached = None
            if text_embedding:
                semantic_cached = find_best_cached_analysis(user_input, text_embedding, STORAGE_FILE, context_snapshot=None, similarity_threshold=similarity_threshold)

            if semantic_cached:
                logging.info("Semantic cache hit: reusing similar cached analysis")
                text_embedding = semantic_cached["original_text_embedding"]
                parsed_objects = semantic_cached["parsed_objects"]
            else:
                logging.info("No suitable cached analysis; calling Ollama")
                ollama_json_output = parse_text_to_objects(user_input)
                if not ollama_json_output:
                    logging.warning("Ollama did not return any structured data.")
                    return
                parsed_objects = parse_json_to_tbot_objects(ollama_json_output)

        
        if parsed_objects:
            logging.info("\n--- Successfully Parsed Objects ---")
            for obj in parsed_objects:
                logging.info(f"- {obj.__class__.__name__}: {obj}")
            logging.info("-----------------------------------")
            
            # Add the new analysis to the knowledge base and save (only if not cached)
            add_analysis_to_knowledge_base(user_input, parsed_objects, text_embedding, STORAGE_FILE)
            knowledge_base = load_knowledge_base(STORAGE_FILE)  # Reload to get updated KB
            
            # Perform mathematical reasoning on current input and entire KB
            perform_mathematical_reasoning(user_input, parsed_objects, text_embedding, knowledge_base)

        else:
            logging.warning("No t_bot_objects could be created from Ollama's output.")

    except Exception as e:
        logging.error(f"An unexpected error occurred during processing: {e}", exc_info=True)


def analyze_multiple_sentences(sentences: List[str], knowledge_base: List[Analysis], similarity_threshold: float = 0.85) -> None:
    """Analyze multiple sentences together to find cross-sentence inferences and contradictions."""
    try:
        logging.info(f"\n--- Analyzing {len(sentences)} sentences for cross-sentence reasoning ---")
        
        sentences_data: List[Tuple[str, List[TBotObject], List[float]]] = []
        
        # Create a context snapshot for the group of sentences to preserve discourse-level meaning
        context_snapshot = hashlib.sha256(" ".join(sentences).encode("utf-8")).hexdigest()

        # Process each sentence with hybrid cache lookup
        for i, sentence in enumerate(sentences, 1):
            logging.info(f"\nProcessing sentence {i}: '{sentence}'")

            # 1) exact-text cache (fast)
            cached = find_analysis_by_text(sentence, STORAGE_FILE)
            if cached:
                logging.info("  Exact-text cache hit: reusing analysis")
                parsed_objects = cached["parsed_objects"]
                embedding = cached["original_text_embedding"]
            else:
                # 2) compute embedding for semantic fallback
                embedding = generate_text_embedding(sentence)
                if embedding:
                    logging.info(f"  Embedding generated (first 5 elements): {embedding[:5]}...")
                else:
                    logging.warning("  Failed to generate embedding.")
                    embedding = []

                # 3) semantic cache with context-aware snapshot
                semantic_cached = None
                if embedding:
                    semantic_cached = find_best_cached_analysis(sentence, embedding, STORAGE_FILE, context_snapshot=context_snapshot, similarity_threshold=similarity_threshold)

                if semantic_cached:
                    logging.info("  Semantic cache hit: reusing similar analysis")
                    parsed_objects = semantic_cached["parsed_objects"]
                    embedding = semantic_cached["original_text_embedding"]
                else:
                    # 4) fallback: call Ollama and parse
                    ollama_json_output = parse_text_to_objects(sentence)
                    if not ollama_json_output:
                        logging.warning(f"  Ollama did not return structured data for this sentence.")
                        parsed_objects = []
                    else:
                        parsed_objects = parse_json_to_tbot_objects(ollama_json_output)
                        if parsed_objects:
                            logging.info(f"  Parsed {len(parsed_objects)} objects")
                            for obj in parsed_objects:
                                logging.info(f"    - {obj.__class__.__name__}: {obj}")

                # Add to knowledge base with context snapshot and a model marker
                add_analysis_to_knowledge_base(sentence, parsed_objects, embedding, STORAGE_FILE, context_snapshot=context_snapshot, model_used="llama3")

            sentences_data.append((sentence, parsed_objects, embedding))
        
        # Reload knowledge base with all new sentences
        knowledge_base = load_knowledge_base(STORAGE_FILE)
        
        # Perform multi-sentence reasoning
        logging.info("\n--- Performing Cross-Sentence Analysis ---")
        result = combine_truths_across_sentences(sentences_data, knowledge_base)
        
        # Display results
        if result.get("inferences"):
            logging.info("\n[CROSS-SENTENCE INFERENCES]")
            for inference in result["inferences"]:
                logging.info(f"  → {inference}")
        
        if result.get("contradictions"):
            logging.warning("\n[CONTRADICTIONS DETECTED]")
            for contra in result["contradictions"]:
                logging.warning(f"  CONFLICT: '{contra['fact1']}' (from: {contra['source1']})")
                logging.warning(f"       vs.  '{contra['fact2']}' (from: {contra['source2']})")
                logging.warning(f"       Reason: {contra['explanation']}")
        
        if not result.get("inferences") and not result.get("contradictions"):
            logging.info("\n(No new cross-sentence inferences or contradictions found)")

    except Exception as e:
        logging.error(f"An error occurred during multi-sentence analysis: {e}", exc_info=True)


def main(similarity_threshold: float = 0.85):
    # Configure logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.info("--- t-bot Application Start ---")

    logging.info("Welcome to t-bot! Your linguistic analysis assistant.")
    logging.info("Ollama will parse your text into structured linguistic objects and generate embeddings.")
    logging.info(f"Analyses will be saved to and loaded from '{STORAGE_FILE}' as a knowledge base.")
    logging.info("Type 'load' to view previously saved analysis.")
    logging.info("Type 'multi' to analyze multiple sentences together.")
    logging.info("Type 'exit' to quit.")
    
    # Try to load existing knowledge base at startup
    knowledge_base: List[Analysis] = load_knowledge_base(STORAGE_FILE)
    if knowledge_base:
        logging.info(f"\nLoaded {len(knowledge_base)} existing analyses into the knowledge base.")

    while True:
        user_input = input("\nEnter text for analysis (or 'load' / 'multi' / 'exit'): ").strip()
        
        if user_input.lower() == 'exit':
            logging.info("Exiting application.")
            break
        elif user_input.lower() == 'load':
            loaded_kb: List[Analysis] = load_knowledge_base(STORAGE_FILE)
            if loaded_kb:
                logging.info("\n--- Loaded Knowledge Base Analyses ---")
                for i, analysis in enumerate(loaded_kb):
                    logging.info(f"\n--- Analysis {i+1} ---")
                    logging.info(f"Original Text: {analysis['original_text']}")
                    # Check if embedding exists and is not empty
                    if analysis.get('original_text_embedding'):
                        logging.info(f"Embedding (first 5 elements): {analysis['original_text_embedding'][:5]}...")
                    else:
                        logging.info("Embedding: Not available")
                    logging.info("Parsed Objects:")
                    for obj in analysis['parsed_objects']:
                        logging.info(f"  - {obj.__class__.__name__}: {obj}")
                logging.info("------------------------------------")
            else:
                logging.warning("Knowledge base is empty.")
            continue
        elif user_input.lower() == 'multi':
            logging.info("\n--- Multi-Sentence Analysis Mode ---")
            sentences: List[str] = []
            logging.info("Enter sentences one per line. Type 'done' when finished.")
            while True:
                sentence = input("Enter sentence (or 'done'): ").strip()
                if sentence.lower() == 'done':
                    break
                if sentence:
                    sentences.append(sentence)
            
            if sentences:
                analyze_multiple_sentences(sentences, knowledge_base, similarity_threshold=similarity_threshold)
            else:
                logging.info("No sentences entered.")
            continue
        # runtime command to change similarity threshold
        if user_input.lower().startswith("set-threshold"):
            parts = user_input.split()
            if len(parts) >= 2:
                try:
                    new_val = float(parts[1])
                    similarity_threshold = new_val
                    logging.info(f"Similarity threshold set to {similarity_threshold}")
                except ValueError:
                    logging.warning("Invalid threshold value. Use a float like: set-threshold 0.9")
            else:
                logging.info(f"Current similarity threshold: {similarity_threshold}")
            continue

        try:
            logging.info("\nAnalyzing text with Ollama...")
            # 1. Generate embedding for the original text
            text_embedding = generate_text_embedding(user_input)
            if text_embedding:
                logging.info(f"Generated Embedding (first 5 elements): {text_embedding[:5]}...")
            else:
                logging.warning("Failed to generate embedding.")
                text_embedding = [] # Ensure embedding is a list

            # 2. Parse text into structured JSON using Ollama
            ollama_json_output = parse_text_to_objects(user_input)
            
            if not ollama_json_output:
                logging.warning("Ollama did not return any structured data.")
                continue

            # 3. Convert structured JSON into t_bot_objects instances
            parsed_objects = parse_json_to_tbot_objects(ollama_json_output)
            
            if parsed_objects:
                logging.info("\n--- Successfully Parsed Objects ---")
                for obj in parsed_objects:
                    logging.info(f"- {obj.__class__.__name__}: {obj}")
                logging.info("-----------------------------------")
                
                # Add the new analysis to the knowledge base and save
                add_analysis_to_knowledge_base(user_input, parsed_objects, text_embedding, STORAGE_FILE)
                knowledge_base = load_knowledge_base(STORAGE_FILE) # Reload to get updated KB
                
                # Perform mathematical reasoning on current input and entire KB
                perform_mathematical_reasoning(user_input, parsed_objects, text_embedding, knowledge_base)

            else:
                logging.warning("No t_bot_objects could be created from Ollama's output.")

        except Exception as e:
            logging.error(f"An unexpected error occurred during processing: {e}", exc_info=True)

        analyze_single_sentence(user_input, knowledge_base, similarity_threshold=similarity_threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="t-bot CLI")
    parser.add_argument("--similarity", "-s", type=float, default=0.85, help="Similarity threshold for semantic cache (0-1)")
    args = parser.parse_args()
    main(similarity_threshold=args.similarity)
