from typing import Union, Optional, List, TypedDict

# --- TypedDicts for to_dict() methods ---

class NounDict(TypedDict):
    _type: str
    name: str
    description: str
    modifiers: List[str]

class VerbDict(TypedDict):
    _type: str
    name: str
    action: str

class AdjectiveDict(TypedDict):
    _type: str
    name: str
    quality: str

class AdverbDict(TypedDict):
    _type: str
    name: str
    manner: str

class PronounDict(TypedDict):
    _type: str
    name: str
    reference: str

class PrepositionDict(TypedDict):
    _type: str
    name: str
    relation: str

class ConjunctionDict(TypedDict):
    _type: str
    name: str
    connection: str

class InterjectionDict(TypedDict):
    _type: str
    name: str
    expression: str

class ArticleDict(TypedDict):
    _type: str
    name: str
    type_: str

class DeterminerDict(TypedDict):
    _type: str
    name: str
    specification: str

class NumeralDict(TypedDict):
    _type: str
    name: str
    quantity: int

class ParticleDict(TypedDict):
    _type: str
    name: str
    function: str

class ModalDict(TypedDict):
    _type: str
    name: str
    mood: str

class ClauseDict(TypedDict):
    _type: str
    subject: str
    predicate: str

class PhraseDict(TypedDict):
    _type: str
    type_: str
    content: str

class LanguageIntentDict(TypedDict):
    _type: str
    intent_type: str
    description: str
    purposes: List[str]

class LanguageContextDict(TypedDict):
    _type: str
    context_type: str
    details: str

class FactsFromTextDict(TypedDict):
    _type: str
    fact: str
    source: str


# --- t_bot_objects Classes ---

class Noun:
    def __init__(self, name: str, description: str, modifiers: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.modifiers = modifiers if modifiers is not None else []

    def __repr__(self):
        return f"Noun(name={self.name}, description={self.description}, modifiers={self.modifiers})"
    
    def to_dict(self) -> NounDict:
        return {"_type": "Noun", "name": self.name, "description": self.description, "modifiers": self.modifiers}

class Verb:
    def __init__(self, name: str, action: str):
        self.name = name
        self.action = action

    def __repr__(self):
        return f"Verb(name={self.name}, action={self.action})"

    def to_dict(self) -> VerbDict:
        return {"_type": "Verb", "name": self.name, "action": self.action}

class Adjective:    
    def __init__(self, name: str, quality: str):
        self.name = name
        self.quality = quality

    def __repr__(self):
        return f"Adjective(name={self.name}, quality={self.quality})"
    
    def to_dict(self) -> AdjectiveDict:
        return {"_type": "Adjective", "name": self.name, "quality": self.quality}

class Adverb:
    def __init__(self, name: str, manner: str):
        self.name = name
        self.manner = manner

    def __repr__(self):
        return f"Adverb(name={self.name}, manner={self.manner})"

    def to_dict(self) -> AdverbDict:
        return {"_type": "Adverb", "name": self.name, "manner": self.manner}

class Pronoun:
    def __init__(self, name: str, reference: str):
        self.name = name
        self.reference = reference

    def __repr__(self):
        return f"Pronoun(name={self.name}, reference={self.reference})"

    def to_dict(self) -> PronounDict:
        return {"_type": "Pronoun", "name": self.name, "reference": self.reference}

class Preposition :
    def __init__(self, name: str, relation: str):
        self.name = name
        self.relation = relation

    def __repr__(self):
        return f"Preposition(name={self.name}, relation={self.relation})"
    
    def to_dict(self) -> PrepositionDict:
        return {"_type": "Preposition", "name": self.name, "relation": self.relation}
    
class Conjunction:
    def __init__(self, name: str, connection: str):
        self.name = name
        self.connection = connection

    def __repr__(self):
        return f"Conjunction(name={self.name}, connection={self.connection})" 
    
    def to_dict(self) -> ConjunctionDict:
        return {"_type": "Conjunction", "name": self.name, "connection": self.connection} 

class Interjection:
    def __init__(self, name: str, expression: str):
        self.name = name
        self.expression = expression

    def __repr__(self):
        return f"Interjection(name={self.name}, expression={self.expression})"
    
    def to_dict(self) -> InterjectionDict:
        return {"_type": "Interjection", "name": self.name, "expression": self.expression}

class Article:
    def __init__(self, name: str, type_: str):
        self.name = name
        self.type_ = type_

    def __repr__(self):
        return f"Article(name={self.name}, type_={self.type_})"
    
    def to_dict(self) -> ArticleDict:
        return {"_type": "Article", "name": self.name, "type_": self.type_}
    
class Determiner:
    def __init__(self, name: str, specification: str):
        self.name = name
        self.specification = specification

    def __repr__(self):
        return f"Determiner(name={self.name}, specification={self.specification})"  
    
    def to_dict(self) -> DeterminerDict:
        return {"_type": "Determiner", "name": self.name, "specification": self.specification}  

class Numeral:
    def __init__(self, name: str, quantity: int):
        self.name = name
        self.quantity = quantity

    def __repr__(self):
        return f"Numeral(name={self.name}, quantity={self.quantity})"
    
    def to_dict(self) -> NumeralDict:
        return {"_type": "Numeral", "name": self.name, "quantity": self.quantity}

class Particle:
    def __init__(self, name: str, function: str):
        self.name = name
        self.function = function

    def __repr__(self):
        return f"Particle(name={self.name}, function={self.function})"   
    
    def to_dict(self) -> ParticleDict:
        return {"_type": "Particle", "name": self.name, "function": self.function}   

class Modal:    
    def __init__(self, name: str, mood: str):
        self.name = name
        self.mood = mood

    def __repr__(self):
        return f"Modal(name={self.name}, mood={self.mood})" 

    def to_dict(self) -> ModalDict:
        return {"_type": "Modal", "name": self.name, "mood": self.mood} 

class Clause:
    def __init__(self, subject: str, predicate: str):
        self.subject = subject
        self.predicate = predicate

    def __repr__(self):
        return f"Clause(subject={self.subject}, predicate={self.predicate})"

    def to_dict(self) -> ClauseDict:
        return {"_type": "Clause", "subject": self.subject, "predicate": self.predicate}

class Phrase:
    def __init__(self, type_: str, content: str):
        self.type_ = type_
        self.content = content

    def __repr__(self):
        return f"Phrase(type_={self.type_}, content={self.content})"
    
    def to_dict(self) -> PhraseDict:
        return {"_type": "Phrase", "type_": self.type_, "content": self.content}
    
class LanguageIntent:
    def __init__(self, intent_type: str, description: str, purposes: Optional[List[str]] = None):
        self.intent_type = intent_type
        self.description = description
        self.purposes = purposes if purposes is not None else []

    def __repr__(self):
        return f"LanguageIntent(intent_type={self.intent_type}, description={self.description}, purposes={self.purposes})"    
    
    def to_dict(self) -> LanguageIntentDict:
        return {"_type": "LanguageIntent", "intent_type": self.intent_type, "description": self.description, "purposes": self.purposes}    
    
class LanguageContext:
    def __init__(self, context_type: str, details: str):
        self.context_type = context_type
        self.details = details 

    def __repr__(self):
        return f"LanguageContext(context_type={self.context_type}, details={self.details})"
    
    def to_dict(self) -> LanguageContextDict:
        return {"_type": "LanguageContext", "context_type": self.context_type, "details": self.details}
    
class FactsFromText:
    def __init__(self, fact: str, source: str):
        self.fact = fact
        self.source = source

    def __repr__(self):
        return f"FactsFromText(fact={self.fact}, source={self.source})"
    
    def to_dict(self) -> FactsFromTextDict:
        return {"_type": "FactsFromText", "fact": self.fact, "source": self.source}

TBotObject = Union[

    Noun, Verb, Adjective, Adverb, Pronoun, Preposition, Conjunction, Interjection,

    Article, Determiner, Numeral, Particle, Modal, Clause, Phrase,

    LanguageIntent, LanguageContext, FactsFromText

]



TBotObjectDict = Union[

    NounDict, VerbDict, AdjectiveDict, AdverbDict, PronounDict, PrepositionDict,

    ConjunctionDict, InterjectionDict, ArticleDict, DeterminerDict, NumeralDict,

    ParticleDict, ModalDict, ClauseDict, PhraseDict, LanguageIntentDict,

    LanguageContextDict, FactsFromTextDict

]
