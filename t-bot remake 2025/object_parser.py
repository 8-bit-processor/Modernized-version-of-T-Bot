from typing import List
from t_bot_objects import (
    Noun, Verb, Adjective, Adverb, Pronoun, Preposition, Conjunction, Interjection,
    Article, Determiner, Numeral, Particle, Modal, Clause, Phrase,
    LanguageIntent, LanguageContext, FactsFromText, TBotObject
)
from ollama_module import OllamaOutputSchema

def parse_json_to_tbot_objects(json_data: OllamaOutputSchema) -> List[TBotObject]:
    """
    Parses a dictionary (expected to be from Ollama's structured JSON output)
    into a list of t_bot_objects instances.
    """
    tbot_objects: List[TBotObject] = []

    # Nouns
    for item in json_data.get("nouns", []):
        tbot_objects.append(Noun(name=item.get("name", ""), description=item.get("description", ""), modifiers=item.get("modifiers", [])))

    # Verbs
    for item in json_data.get("verbs", []):
        tbot_objects.append(Verb(name=item.get("name", ""), action=item.get("action", "")))

    # Adjectives
    for item in json_data.get("adjectives", []):
        tbot_objects.append(Adjective(name=item.get("name", ""), quality=item.get("quality", "")))

    # Adverbs
    for item in json_data.get("adverbs", []):
        tbot_objects.append(Adverb(name=item.get("name", ""), manner=item.get("manner", "")))

    # Pronouns
    for item in json_data.get("pronouns", []):
        tbot_objects.append(Pronoun(name=item.get("name", ""), reference=item.get("reference", "")))

    # Prepositions
    for item in json_data.get("prepositions", []):
        tbot_objects.append(Preposition(name=item.get("name", ""), relation=item.get("relation", "")))
        
    # Conjunctions
    for item in json_data.get("conjunctions", []):
        tbot_objects.append(Conjunction(name=item.get("name", ""), connection=item.get("connection", "")))

    # Interjections
    for item in json_data.get("interjections", []):
        tbot_objects.append(Interjection(name=item.get("name", ""), expression=item.get("expression", "")))

    # Articles
    for item in json_data.get("articles", []):
        tbot_objects.append(Article(name=item.get("name", ""), type_=item.get("type_", "")))

    # Determiners
    for item in json_data.get("determiners", []):
        tbot_objects.append(Determiner(name=item.get("name", ""), specification=item.get("specification", "")))

    # Numerals
    for item in json_data.get("numerals", []):
        tbot_objects.append(Numeral(name=item.get("name", ""), quantity=item.get("quantity", 0)))

    # Particles
    for item in json_data.get("particles", []):
        tbot_objects.append(Particle(name=item.get("name", ""), function=item.get("function", "")))

    # Modals
    for item in json_data.get("modals", []):
        tbot_objects.append(Modal(name=item.get("name", ""), mood=item.get("mood", "")))

    # Clauses
    for item in json_data.get("clauses", []):
        tbot_objects.append(Clause(subject=item.get("subject", ""), predicate=item.get("predicate", "")))

    # Phrases
    for item in json_data.get("phrases", []):
        tbot_objects.append(Phrase(type_=item.get("type_", ""), content=item.get("content", "")))
    
    # LanguageIntent
    intent_data = json_data.get("language_intent")
    if intent_data:
        tbot_objects.append(
            LanguageIntent(
                intent_type=intent_data.get("intent_type", ""),
                description=intent_data.get("description", "")
            )
        )
    
    # LanguageContext
    context_data = json_data.get("language_context")
    if context_data:
        tbot_objects.append(
            LanguageContext(
                context_type=context_data.get("context_type", ""),
                details=context_data.get("details", "")
            )
        )

    # FactsFromText
    for item in json_data.get("facts_from_text", []):
        tbot_objects.append(FactsFromText(fact=item.get("fact", ""), source=item.get("source", "")))

    return tbot_objects

