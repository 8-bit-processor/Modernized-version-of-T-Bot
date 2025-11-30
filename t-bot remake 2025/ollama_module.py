import json
import httpx
import logging
from typing import List, TypedDict, Optional, Mapping, Any, Sequence, cast
from ollama import chat as _chat, embeddings, ChatResponse, EmbeddingsResponse  # type: ignore[reportUnknownVariableType]

# --- Define TypedDicts for the Ollama JSON Schema ---
class NounSchema(TypedDict):
    name: str
    description: str
    modifiers: List[str]

class VerbSchema(TypedDict):
    name: str
    action: str

class AdjectiveSchema(TypedDict):
    name: str
    quality: str

class AdverbSchema(TypedDict):
    name: str
    manner: str

class PronounSchema(TypedDict):
    name: str
    reference: str

class PrepositionSchema(TypedDict):
    name: str
    relation: str

class ConjunctionSchema(TypedDict):
    name: str
    connection: str

class InterjectionSchema(TypedDict):
    name: str
    expression: str

class ArticleSchema(TypedDict):
    name: str
    type_: str

class DeterminerSchema(TypedDict):
    name: str
    specification: str

class NumeralSchema(TypedDict):
    name: str
    quantity: int

class ParticleSchema(TypedDict):
    name: str
    function: str

class ModalSchema(TypedDict):
    name: str
    mood: str

class ClauseSchema(TypedDict):
    subject: str
    predicate: str

class PhraseSchema(TypedDict):
    type_: str
    content: str

class LanguageIntentSchema(TypedDict):
    intent_type: str
    description: str

class LanguageContextSchema(TypedDict):
    context_type: str
    details: str

class FactsFromTextSchema(TypedDict):
    fact: str
    source: str

class OllamaOutputSchema(TypedDict):
    """
    TypedDict representing the full JSON schema expected from Ollama's linguistic analysis.
    """
    nouns: List[NounSchema]
    verbs: List[VerbSchema]
    adjectives: List[AdjectiveSchema]
    adverbs: List[AdverbSchema]
    pronouns: List[PronounSchema]
    prepositions: List[PrepositionSchema]
    conjunctions: List[ConjunctionSchema]
    interjections: List[InterjectionSchema]
    articles: List[ArticleSchema]
    determiners: List[DeterminerSchema]
    numerals: List[NumeralSchema]
    particles: List[ParticleSchema]
    modals: List[ModalSchema]
    clauses: List[ClauseSchema]
    phrases: List[PhraseSchema]
    language_intent: LanguageIntentSchema
    language_context: LanguageContextSchema
    facts_from_text: List[FactsFromTextSchema]

# --- TypedDicts for Ollama API Responses ---
class OllamaMessage(TypedDict):
    role: str
    content: str


# Top-level typed wrapper for the non-streaming chat call.
def chat_sync(model: str = "", messages: Optional[Sequence[Mapping[str, Any]]] = None, **kwargs: Any) -> ChatResponse:
    """Call Ollama's chat in synchronous (non-streaming) mode with an explicit return type.

    Import is `_chat` because the underlying callable has overloaded signatures; using this wrapper
    makes the intended (sync) overload explicit for static type checkers and IDEs.
    """
    return _chat(model=model, messages=messages, stream=False, **kwargs)


SYSTEM_PROMPT = """You are an expert linguistic analysis system. Your task is to analyze the provided text and extract its core linguistic components, intent, and any factual information.
Respond ONLY with a JSON object that strictly adheres to the following structure.
Every key in the JSON schema must be present in your response, even if its value is an empty list.

Crucially, you must identify any statement that represents a logical implication and structure it as a fact.
For example, a sentence like "If it is raining, then the ground is wet" must be extracted into the `facts_from_text` array as:
`{"fact": "IF it is raining THEN the ground is wet", "source": "input_text"}`.

JSON Schema:
{
    "nouns": [
        {"name": "string", "description": "string", "modifiers": ["string"]}
    ],
    "verbs": [
        {"name": "string", "action": "string"}
    ],
    "adjectives": [
        {"name": "string", "quality": "string"}
    ],
    "adverbs": [
        {"name": "string", "manner": "string"}
    ],
    "pronouns": [
        {"name": "string", "reference": "string"}
    ],
    "prepositions": [
        {"name": "string", "relation": "string"}
    ],
    "conjunctions": [
        {"name": "string", "connection": "string"}
    ],
    "interjections": [
        {"name": "string", "expression": "string"}
    ],
    "articles": [
        {"name": "string", "type_": "string"}
    ],
    "determiners": [
        {"name": "string", "specification": "string"}
    ],
    "numerals": [
        {"name": "string", "quantity": "integer"}
    ],
    "particles": [
        {"name": "string", "function": "string"}
    ],
    "modals": [
        {"name": "string", "mood": "string"}
    ],
    "clauses": [
        {"subject": "string", "predicate": "string"}
    ],
    "phrases": [
        {"type_": "string", "content": "string"}
    ],
    "language_intent": {
        "intent_type": "string",
        "description": "string"
    },
    "language_context": {
        "context_type": "string",
        "details": "string"
    },
    "facts_from_text": [
        {"fact": "string (For implications, use 'IF [antecedent] THEN [consequent]' format)", "source": "input_text"}
    ]
}
Analyze the input text and extract instances for each category. For 'facts_from_text', the source should always be "input_text".
Your response MUST be a single JSON object and nothing else.
"""

def parse_text_to_objects(text: str) -> OllamaOutputSchema:
    messages: List[OllamaMessage] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text}
    ]
    try:
        response: ChatResponse = chat_sync(model="llama3", messages=messages, format="json")
    except httpx.RequestError as e:
        logging.error(f"Error connecting to Ollama server for chat: {e}", exc_info=True)
        # Return an empty schema
        return {
            "nouns": [], "verbs": [], "adjectives": [], "adverbs": [], "pronouns": [],
            "prepositions": [], "conjunctions": [], "interjections": [], "articles": [],
            "determiners": [], "numerals": [], "particles": [], "modals": [], "clauses": [],
            "phrases": [], "language_intent": {"intent_type": "", "description": ""},
            "language_context": {"context_type": "", "details": ""}, "facts_from_text": []
        }
    except Exception as e:
        logging.error(f"An unexpected error occurred during Ollama chat: {e}", exc_info=True)
        return {
            "nouns": [], "verbs": [], "adjectives": [], "adverbs": [], "pronouns": [],
            "prepositions": [], "conjunctions": [], "interjections": [], "articles": [],
            "determiners": [], "numerals": [], "particles": [], "modals": [], "clauses": [],
            "phrases": [], "language_intent": {"intent_type": "", "description": ""},
            "language_context": {"context_type": "", "details": ""}, "facts_from_text": []
        }
    
    logging.debug(f"Ollama raw response: {response}")

    try:
        parsed_data = cast(OllamaOutputSchema, json.loads(response["message"]["content"]))
        if "nouns" not in parsed_data:
            logging.warning("Ollama output missing 'nouns' key, returning empty schema.")
            return cast(OllamaOutputSchema, {
                "nouns": [], "verbs": [], "adjectives": [], "adverbs": [], "pronouns": [],
                "prepositions": [], "conjunctions": [], "interjections": [], "articles": [],
                "determiners": [], "numerals": [], "particles": [], "modals": [], "clauses": [],
                "phrases": [], "language_intent": {"intent_type": "", "description": ""},
                "language_context": {"context_type": "", "details": ""}, "facts_from_text": []
            })
        return parsed_data
    except (json.JSONDecodeError, KeyError) as e:
        logging.error(f"Error processing Ollama chat response: {e}", exc_info=True)
        logging.error(f"Ollama's raw response content: {response.get('message', {}).get('content')}")
        return cast(OllamaOutputSchema, {
            "nouns": [], "verbs": [], "adjectives": [], "adverbs": [], "pronouns": [],
            "prepositions": [], "conjunctions": [], "interjections": [], "articles": [],
            "determiners": [], "numerals": [], "particles": [], "modals": [], "clauses": [],
            "phrases": [], "language_intent": {"intent_type": "", "description": ""},
            "language_context": {"context_type": "", "details": ""}, "facts_from_text": []
        })

def generate_text_embedding(text: str, model: str = "nomic-embed-text") -> List[float]:
    """
    Generates a numerical embedding (vector) for the given text using Ollama.
    """
    try:
        response: EmbeddingsResponse = embeddings(model=model, prompt=text)
        return response.get("embedding", [])
    except httpx.RequestError as e:
        logging.error(f"Error connecting to Ollama server for embeddings: {e}", exc_info=True)
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred during Ollama embeddings generation: {e}", exc_info=True)
        return []