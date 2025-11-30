import logging
import re
from typing import List, Tuple, Dict, Set, Any, Optional

import numpy as np

from ollama_module import generate_text_embedding
from persistence_manager import Analysis
from t_bot_objects import FactsFromText, LanguageIntent, TBotObject


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculates the cosine similarity between two vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        logging.debug("One or both vectors have zero norm.")
        return 0.0
    similarity = dot_product / (norm_v1 * norm_v2)
    logging.debug(f"Cosine similarity: {similarity:.4f}")
    return similarity


def evaluate_proposition(
    proposition_text: str,
    current_objects: List[TBotObject],
    current_embedding: Optional[List[float]],
    knowledge_base: List[Analysis],
    proposition_similarity_threshold: float = 0.7,
) -> bool:
    """
    Evaluates if a given proposition is 'True' based on semantic similarity
    to the current context and the knowledge base.
    """
    logging.info(f"  - Evaluating proposition: '{proposition_text}'")
    prop_embedding = generate_text_embedding(proposition_text)
    if not prop_embedding:
        logging.warning(
            f"    Could not generate embedding for proposition '{proposition_text}'. Assuming False."
        )
        return False

    if current_embedding:
        sim_with_current = cosine_similarity(prop_embedding, current_embedding)
        logging.debug(f"    Similarity with current input embedding: {sim_with_current:.2f}")
        if sim_with_current >= proposition_similarity_threshold:
            logging.info(f"    -> True (similar to current input: {sim_with_current:.2f})")
            return True

    for obj in current_objects:
        if isinstance(obj, FactsFromText):
            fact_embedding = generate_text_embedding(obj.fact)
            if fact_embedding:
                sim_with_fact = cosine_similarity(prop_embedding, fact_embedding)
                logging.debug(f"    Similarity with current fact '{obj.fact}': {sim_with_fact:.2f}")
                if sim_with_fact >= proposition_similarity_threshold:
                    logging.info(
                        f"    -> True (similar to fact: '{obj.fact}' ({sim_with_fact:.2f}))"
                    )
                    return True

    for i, analysis in enumerate(knowledge_base):
        if analysis.get("original_text_embedding"):
            sim_with_kb_text = cosine_similarity(
                prop_embedding, analysis["original_text_embedding"]
            )
            logging.debug(f"    Similarity with KB entry {i} text: {sim_with_kb_text:.2f}")
            if sim_with_kb_text >= proposition_similarity_threshold:
                logging.info(
                    f"    -> True (similar to KB text: '{analysis['original_text']}' ({sim_with_kb_text:.2f}))"
                )
                return True
        for obj in analysis.get("parsed_objects", []):
            if isinstance(obj, FactsFromText):
                kb_fact_embedding = generate_text_embedding(obj.fact)
                if kb_fact_embedding:
                    sim_with_kb_fact = cosine_similarity(prop_embedding, kb_fact_embedding)
                    logging.debug(f"    Similarity with KB fact '{obj.fact}': {sim_with_kb_fact:.2f}")
                    if sim_with_kb_fact >= proposition_similarity_threshold:
                        logging.info(
                            f"    -> True (similar to KB fact: '{obj.fact}' ({sim_with_kb_fact:.2f}))"
                        )
                        return True

    logging.info(f"    -> False (no sufficient similarity found for '{proposition_text}')")
    return False


def kleene_truth_tables(p: bool, q: bool, operator: str) -> bool:
    """A basic truth table evaluator for propositional logic operators."""
    if operator == "AND":
        return p and q
    if operator == "OR":
        return p or q
    if operator == "NOT":
        return not p
    if operator == "IMPLIES":
        return (not p) or q  # P -> Q
    if operator == "EQUIV":
        return p == q  # P <-> Q
    raise ValueError(f"Unknown operator: {operator}")


def modus_ponens(p_implies_q_is_true: bool, p_is_true: bool) -> bool:
    """If P -> Q is true and P is true, then Q is true."""
    if p_implies_q_is_true and p_is_true:
        return True  # Q is true
    return False


def modus_tollens(p_implies_q_is_true: bool, not_q_is_true: bool) -> bool:
    """If P -> Q is true and not Q is true, then not P is true."""
    if p_implies_q_is_true and not_q_is_true:
        return True  # Not P is true
    return False


def extract_implication_rules_from_facts(
    current_objects: List[TBotObject], knowledge_base: List[Analysis]
) -> List[Tuple[str, str]]:
    """
    Scans FactsFromText objects for patterns that suggest an explicit implication rule
    like "If P then Q", "P implies Q", "X leads to Y", etc.
    Returns a list of (antecedent_text, consequent_text) tuples.
    """
    implication_rules: List[Tuple[str, str]] = []

    implication_pattern = re.compile(
        r"(?:if\s*(.+?)\s*then\s*(.+))|"  # Pattern 1: if X, then Y -> Groups (1, 2)
        r"(.+?)\s+implies\s+(.+)|"  # Pattern 2: X implies Y -> Groups (3, 4)
        r"(.+?)\s+leads to\s+(.+)|"  # Pattern 3: X leads to Y -> Groups (5, 6)
        r"(.+?)\s+causes\s+(.+)|"  # Pattern 4: X causes Y -> Groups (7, 8)
        r"(.+?)\s+results in\s+(.+)|"  # Pattern 5: X results in Y -> Groups (9, 10)
        r"(.+?)\s+will result in\s+(.+)|"  # Pattern 6: X will result in Y -> Groups (11, 12)
        r"(.+?)\s+is a consequence of\s+(.+)|"  # Pattern 7: Y is a consequence of X -> Groups (14, 13)
        r"(.+?)\s+when\s+(.+)|"  # Pattern 8: Y when X -> Groups (16, 15)
        r"(.+?)\s+if\s+(.+)",  # Pattern 9: Y if X -> Groups (18, 17)
        re.IGNORECASE,
    )

    sources: List[str] = []
    for obj in current_objects:
        if isinstance(obj, FactsFromText):
            sources.append(obj.fact)
    for analysis in knowledge_base:
        for obj in analysis.get("parsed_objects", []):
            if isinstance(obj, FactsFromText):
                sources.append(obj.fact)

    logging.debug(f"Scanning {len(sources)} facts for implication rules.")

    for fact in sources:
        match = implication_pattern.search(fact)
        if match:
            rule = None
            if match.group(1) and match.group(2):
                rule = (match.group(1).strip(), match.group(2).strip())
            elif match.group(3) and match.group(4):
                rule = (match.group(3).strip(), match.group(4).strip())
            elif match.group(5) and match.group(6):
                rule = (match.group(5).strip(), match.group(6).strip())
            elif match.group(7) and match.group(8):
                rule = (match.group(7).strip(), match.group(8).strip())
            elif match.group(9) and match.group(10):
                rule = (match.group(9).strip(), match.group(10).strip())
            elif match.group(11) and match.group(12):
                rule = (match.group(11).strip(), match.group(12).strip())
            elif match.group(13) and match.group(14):
                rule = (match.group(14).strip(), match.group(13).strip())
            elif match.group(15) and match.group(16):
                rule = (match.group(16).strip(), match.group(15).strip())
            elif match.group(17) and match.group(18):
                rule = (match.group(18).strip(), match.group(17).strip())

            if rule:
                logging.debug(f"  Found rule in fact '{fact}': IF '{rule[0]}' THEN '{rule[1]}'")
                implication_rules.append(rule)

    unique_rules: List[Tuple[str, str]] = list(set(implication_rules))
    logging.info(f"Found {len(unique_rules)} unique explicit implication rules.")
    return unique_rules


def perform_mathematical_reasoning(
    current_text: str,
    current_objects: List[TBotObject],
    current_embedding: Optional[List[float]],
    knowledge_base: List[Analysis],
    kb_similarity_threshold: float = 0.7,
    prop_similarity_threshold: float = 0.7,
) -> str:
    """
    Performs mathematical reasoning based on current input and the knowledge base.
    Dynamically extracts propositions and applies propositional logic rules.
    """
    logging.info("\n--- Performing Mathematical Reasoning ---")
    logging.info(f"Analyzing current input: '{current_text}'")

    similar_analyses: List[Dict[str, Any]] = []
    if knowledge_base and current_embedding:
        for i, kb_entry in enumerate(knowledge_base):
            if kb_entry.get("original_text_embedding") and kb_entry["original_text"] != current_text:
                similarity = cosine_similarity(
                    current_embedding, kb_entry["original_text_embedding"]
                )
                if similarity >= kb_similarity_threshold:
                    logging.debug(f"KB entry {i} is similar (sim: {similarity:.2f})")
                    similar_analyses.append({"analysis": kb_entry, "similarity": similarity})
        similar_analyses.sort(key=lambda x: x["similarity"], reverse=True)
        if similar_analyses:
            logging.info(
                f"\nFound {len(similar_analyses)} similar contexts (similarity >= {kb_similarity_threshold})."
            )

    logging.info("\n--- Dynamically Applying Propositional Logic Rules ---")

    candidate_propositions_texts: Set[str] = set()

    candidate_propositions_texts.add(current_text)
    for obj in current_objects:
        if isinstance(obj, FactsFromText):
            candidate_propositions_texts.add(obj.fact)
        elif isinstance(obj, LanguageIntent):
            candidate_propositions_texts.add(obj.description)

    for sa in similar_analyses:
        candidate_propositions_texts.add(sa["analysis"]["original_text"])
        for obj in sa["analysis"]["parsed_objects"]:
            if isinstance(obj, FactsFromText):
                candidate_propositions_texts.add(obj.fact)
            elif isinstance(obj, LanguageIntent):
                candidate_propositions_texts.add(obj.description)

    candidate_propositions_list: List[str] = list(candidate_propositions_texts)
    logging.info(f"\nGenerated {len(candidate_propositions_list)} unique candidate propositions.")
    for i, cp in enumerate(candidate_propositions_list):
        logging.debug(f"  Candidate {i+1}: '{cp}'")

    explicit_implication_rules = extract_implication_rules_from_facts(
        current_objects, knowledge_base
    )

    inferred_conclusions: Set[str] = set()
    evaluated_propositions: Dict[str, bool] = {}

    def get_eval(prop_text: str) -> bool:
        if prop_text not in evaluated_propositions:
            logging.debug(f"Caching evaluation for: '{prop_text}'")
            evaluated_propositions[prop_text] = evaluate_proposition(
                prop_text,
                current_objects,
                current_embedding,
                knowledge_base,
                prop_similarity_threshold,
            )
        return evaluated_propositions[prop_text]

    if explicit_implication_rules:
        logging.info("\n--- Applying Modus Ponens/Tollens using Explicit Implication Rules ---")
        for ant_text, con_text in explicit_implication_rules:
            logging.debug(f"Testing rule: IF '{ant_text}' THEN '{con_text}'")
            p_ant_is_true = get_eval(ant_text)
            q_con_is_true = get_eval(con_text)
            # Evaluate the implication itself as true (it was explicitly stated)
            p_implies_q_is_true = True

            # Modus Ponens: IF (rule is true AND antecedent is true) THEN consequent must be true
            if modus_ponens(p_implies_q_is_true, p_ant_is_true):
                conclusion = f"Consequent ('{con_text}') inferred TRUE via Modus Ponens (from explicit rule: If '{ant_text}' then '{con_text}')"
                logging.info(f"  NEW INFERENCE: {conclusion}")
                inferred_conclusions.add(conclusion)

            # Modus Tollens: IF (rule is true AND consequent is false) THEN antecedent must be false
            if modus_tollens(p_implies_q_is_true, not q_con_is_true):
                conclusion = f"Antecedent ('{ant_text}') inferred FALSE via Modus Tollens (from explicit rule and NOT '{con_text}')"
                logging.info(f"  NEW INFERENCE: {conclusion}")
                inferred_conclusions.add(conclusion)

    if inferred_conclusions:
        logging.info("\n--- Inferred Conclusions ---")
        for conclusion in inferred_conclusions:
            logging.info(f"- {conclusion}")
    else:
        logging.info("\nNo new conclusions inferred via dynamic propositional logic for this input.")

    logging.info("\n---------------------------------------------")
    return "Mathematical reasoning completed."


def detect_contradiction(
    prop1_text: str,
    prop1_truth: bool,
    prop2_text: str,
    prop2_truth: bool,
    similarity_threshold: float = 0.75,
) -> Tuple[bool, str]:
    """
    Detect if two propositions are semantically similar but have conflicting truth values.
    Returns (is_contradiction, explanation).
    """
    logging.debug(f"Checking for contradiction between: '{prop1_text}' ({prop1_truth}) and '{prop2_text}' ({prop2_truth})")

    emb1 = generate_text_embedding(prop1_text)
    emb2 = generate_text_embedding(prop2_text)

    if not emb1 or not emb2:
        logging.debug("Could not generate embeddings for contradiction check.")
        return False, ""

    similarity = cosine_similarity(emb1, emb2)
    logging.debug(f"Semantic similarity between propositions: {similarity:.4f}")

    # If semantically similar but truth values differ → contradiction
    if similarity >= similarity_threshold and prop1_truth != prop2_truth:
        explanation = f"CONTRADICTION DETECTED: '{prop1_text}' is {prop1_truth} but '{prop2_text}' is {prop2_truth} (similarity: {similarity:.2f})"
        logging.warning(explanation)
        return True, explanation

    return False, ""


def combine_truths_across_sentences(
    sentences_data: List[Tuple[str, List[TBotObject], List[float]]],
    knowledge_base: List[Analysis],
    similarity_threshold: float = 0.75,
) -> Dict[str, Any]:
    """
    Combine facts from multiple sentences, find new inferences and contradictions.

    Args:
        sentences_data: List of (text, parsed_objects, embedding) tuples
        knowledge_base: Full KB for context
        similarity_threshold: Threshold for contradiction detection

    Returns:
        {
            "inferences": [...],      # New truths derived from multiple sentences
            "contradictions": [...],  # Detected conflicts
            "combined_facts": [...]   # Merged fact set
        }
    """
    logging.info("\n--- Combining Truths Across Multiple Sentences ---")
    all_facts: List[Tuple[str, str]] = []
    inferences: Set[str] = set()
    contradictions: List[Dict[str, Any]] = []

    # Collect all facts from all sentences
    logging.info("Extracting facts from all sentences...")
    for text, objects, _ in sentences_data:
        for obj in objects:
            if isinstance(obj, FactsFromText):
                all_facts.append((obj.fact, text))
                logging.debug(f"  Collected fact: '{obj.fact}' (from: '{text}')")

    logging.info(f"Total facts collected: {len(all_facts)}")

    # Evaluate truth value for each fact
    fact_truths: Dict[str, bool] = {}
    for fact, _ in all_facts:
        # Use a simplified evaluation without current_objects (we're across sentences)
        truth = evaluate_proposition(fact, [], None, knowledge_base)
        fact_truths[fact] = truth

    # Check for contradictions between any pair of facts
    logging.info("\nScanning for contradictions...")
    for i, (fact1, src1) in enumerate(all_facts):
        for fact2, src2 in all_facts[i + 1 :]:
            truth1 = fact_truths[fact1]
            truth2 = fact_truths[fact2]

            is_contra, msg = detect_contradiction(fact1, truth1, fact2, truth2, similarity_threshold)
            if is_contra:
                contradictions.append(
                    {
                        "fact1": fact1,
                        "source1": src1,
                        "fact2": fact2,
                        "source2": src2,
                        "explanation": msg,
                    }
                )

    if contradictions:
        logging.info(f"\nFound {len(contradictions)} contradiction(s)")
    else:
        logging.info("\nNo contradictions detected.")

    # Chain implications across sentences
    logging.info("\nSearching for cross-sentence implications...")
    implication_rules = extract_implication_rules_from_facts(
        [obj for _, objs, _ in sentences_data for obj in objs], knowledge_base
    )

    for ant_text, con_text in implication_rules:
        ant_true = evaluate_proposition(ant_text, [], None, knowledge_base)
        con_true = evaluate_proposition(con_text, [], None, knowledge_base)

        if ant_true and not con_true:
            inference = (
                f"Cross-sentence inference: '{ant_text}' is TRUE, and rule IF '{ant_text}' THEN '{con_text}' → "
                f"Conclude '{con_text}' should be TRUE"
            )
            inferences.add(inference)
            logging.info(f"  NEW INFERENCE: {inference}")

    if inferences:
        logging.info(f"\nFound {len(inferences)} cross-sentence inference(s)")
    else:
        logging.info("\nNo new cross-sentence inferences found.")

    result: Dict[str, Any] = {
        "inferences": list(inferences),
        "contradictions": contradictions,
        "combined_facts": all_facts,
        "fact_truths": fact_truths,
    }

    logging.info("\n--- End Multi-Sentence Analysis ---")
    return result