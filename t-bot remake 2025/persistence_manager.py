import json
import os
import logging
from typing import List, TypedDict, cast, Dict, Any, Type, Optional
from t_bot_objects import (
    Noun, Verb, Adjective, Adverb, Pronoun, Preposition, Conjunction, Interjection,
    Article, Determiner, Numeral, Particle, Modal, Clause, Phrase,
    LanguageIntent, LanguageContext, FactsFromText, TBotObject, TBotObjectDict
)

# --- Type Definitions ---
class Analysis(TypedDict):
    """A TypedDict to represent a single analysis entry in the knowledge base."""
    original_text: str
    original_text_embedding: List[float]
    parsed_objects: List[TBotObject]
    context_snapshot: Optional[str]
    timestamp: Optional[str]
    model_used: Optional[str]

class SerializableAnalysis(TypedDict):
    """A TypedDict for the JSON-serializable version of an analysis."""
    original_text: str
    original_text_embedding: List[float]
    parsed_objects: List[TBotObjectDict]
    context_snapshot: Optional[str]
    timestamp: Optional[str]
    model_used: Optional[str]


# --- Class Mapping for Deserialization ---
CLASS_MAPPING: Dict[str, Type[TBotObject]] = {
    "Noun": Noun, "Verb": Verb, "Adjective": Adjective, "Adverb": Adverb,
    "Pronoun": Pronoun, "Preposition": Preposition, "Conjunction": Conjunction,
    "Interjection": Interjection, "Article": Article, "Determiner": Determiner,
    "Numeral": Numeral, "Particle": Particle, "Modal": Modal, "Clause": Clause,
    "Phrase": Phrase, "LanguageIntent": LanguageIntent, "LanguageContext": LanguageContext,
    "FactsFromText": FactsFromText,
}

def _reconstruct_tbot_objects(parsed_objects_data: List[Dict[str, Any]]) -> List[TBotObject]:
    """Helper to reconstruct t_bot_objects from their dictionary representations."""
    reconstructed_objects: List[TBotObject] = []
    for item in parsed_objects_data:
        obj_type = item.get("_type")
        if isinstance(obj_type, str) and obj_type in CLASS_MAPPING:
            item_copy = item.copy()
            del item_copy["_type"]
            try:
                reconstructed_objects.append(CLASS_MAPPING[obj_type](**item_copy))
            except TypeError as e:
                logging.error(f"Error creating object of type {obj_type} with data {item_copy}: {e}", exc_info=True)
        else:
            logging.warning(f"Unknown object type or missing '_type' key in data: {item}")
    return reconstructed_objects


def find_analysis_by_text(original_text: str, file_path: str) -> Optional[Analysis]:
    """Search the knowledge base for an analysis with the exact `original_text`.

    This provides a simple cache: if the exact input text was analyzed before,
    callers can reuse the stored embedding and parsed objects instead of
    calling Ollama again.
    """
    try:
        kb = load_knowledge_base(file_path)
    except Exception:
        return None

    for entry in kb:
        if entry.get("original_text") == original_text:
            return entry
    return None


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors without numpy."""
    try:
        dot = 0.0
        norm1 = 0.0
        norm2 = 0.0
        for a, b in zip(vec1, vec2):
            dot += a * b
            norm1 += a * a
            norm2 += b * b
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / ((norm1 ** 0.5) * (norm2 ** 0.5))
    except Exception:
        return 0.0


def find_best_cached_analysis(original_text: str, embedding: List[float], file_path: str, context_snapshot: Optional[str] = None, similarity_threshold: float = 0.85) -> Optional[Analysis]:
    """Hybrid cache lookup:

    1. Exact text + context match (if context provided)
    2. Exact text match
    3. Semantic (embedding) nearest neighbor with threshold
    """
    try:
        kb = load_knowledge_base(file_path)
    except Exception:
        return None

    # 1) exact text + context
    if context_snapshot is not None:
        for entry in kb:
            if entry.get("original_text") == original_text and entry.get("context_snapshot") == context_snapshot:
                return entry

    # 2) exact text only
    for entry in kb:
        if entry.get("original_text") == original_text:
            return entry

    # 3) semantic fallback
    best: Optional[Analysis] = None
    best_score = 0.0
    for entry in kb:
        cached_emb = entry.get("original_text_embedding")
        if not cached_emb:
            continue
        sim = _cosine_similarity(embedding, cached_emb)
        if sim > best_score:
            best_score = sim
            best = entry

    if best is not None and best_score >= similarity_threshold:
        return best

    return None

def load_knowledge_base(file_path: str) -> List[Analysis]:
    """
    Loads the entire knowledge base from a JSON file.
    Each analysis is a dictionary conforming to the Analysis TypedDict structure.
    """
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        logging.info(f"Knowledge base file not found or empty: {file_path}. Initializing empty KB.")
        return []

    knowledge_base: List[Analysis] = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = cast(List[Dict[str, Any]], json.load(f))
            
            for analysis_data in data:
                reconstructed_objects = _reconstruct_tbot_objects(analysis_data.get("parsed_objects", []))
                analysis: Analysis = {
                    "original_text": analysis_data.get("original_text", ""),
                    "original_text_embedding": analysis_data.get("original_text_embedding", []),
                    "parsed_objects": reconstructed_objects,
                    "context_snapshot": analysis_data.get("context_snapshot"),
                    "timestamp": analysis_data.get("timestamp"),
                    "model_used": analysis_data.get("model_used"),
                }
                knowledge_base.append(analysis)
        logging.info(f"Successfully loaded {len(knowledge_base)} analyses from KB: {file_path}")
        return knowledge_base
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from KB file {file_path}: {e}", exc_info=True)
        return []
    except IOError as e:
        logging.error(f"Error loading KB from file {file_path}: {e}", exc_info=True)
        return []

def add_analysis_to_knowledge_base(original_text: str, objects: List[TBotObject], embedding: List[float], file_path: str, context_snapshot: Optional[str] = None, model_used: Optional[str] = None):
    """
    Adds a new analysis to the knowledge base and saves the entire updated base to a JSON file.
    """
    # Load the KB, which contains live TBotObject instances
    knowledge_base = load_knowledge_base(file_path)
    
    # Create the new analysis with live objects
    new_analysis: Analysis = {
        "original_text": original_text,
        "original_text_embedding": embedding,
        "parsed_objects": objects,
        "context_snapshot": context_snapshot,
        "timestamp": None,
        "model_used": model_used,
    }
    
    # Append the new analysis to the list
    knowledge_base.append(new_analysis)

    # Create a fully serializable version of the entire knowledge base for JSON storage
    serializable_knowledge_base: List[SerializableAnalysis] = []
    for analysis in knowledge_base:
        serializable_analysis: SerializableAnalysis = {
            "original_text": analysis["original_text"],
            "original_text_embedding": analysis["original_text_embedding"],
            "parsed_objects": [obj.to_dict() for obj in analysis["parsed_objects"]],
            "context_snapshot": analysis.get("context_snapshot"),
            "timestamp": analysis.get("timestamp"),
            "model_used": analysis.get("model_used"),
        }
        serializable_knowledge_base.append(serializable_analysis)

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_knowledge_base, f, indent=4)
        logging.info(f"Successfully saved knowledge base. Total analyses: {len(serializable_knowledge_base)}")
    except IOError as e:
        logging.error(f"Error writing to knowledge base file {file_path}: {e}", exc_info=True)

