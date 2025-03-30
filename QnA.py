#   MIT License  
#   Copyright (c) 2025 [Kshitij Dalvi] kshitjdalvi2022@gmail.com 

#  Permission is granted to use, copy, and modify this software **with attribution**.  
#  Redistribution without proper credit is prohibited.  
#  Unauthorized commercial use requires a license agreement.  



import spacy
import random

# Load Transformer-based model
nlp = spacy.load("en_core_web_trf")

def generate_questions(text, max_questions=5):
    doc = nlp(text)
    questions = []
    seen_entities = set()
    question_templates = {
        "principles": "What are the key principles behind {topic}?",
        "applications": "What are real-world applications of {topic}?",
        "misconceptions": "What are common misconceptions about {topic}?",
        "historical_significance": "What historical or political significance does {topic} have?",
        "scientific_contribution": "How did {topic} contribute to further advancements?",
        "global_impact": "What is the impact of '{topic}' ?"
    }

    for sent in doc.sents:
        entities = {ent.label_: ent.text for ent in sent.ents}
        question_list = []
        main_concepts = []
        
        # 1️⃣ ENTITY-BASED QUESTIONS (Ask once per entity)
        if "PERSON" in entities and entities["PERSON"] not in seen_entities:
            question_list.append(f"Who is/was {entities['PERSON']} and what were their contributions?")
            seen_entities.add(entities["PERSON"])

        if "DATE" in entities:
            question_list.append(f"What significant event occurred in {entities['DATE']} and why was it important?")

        if "GPE" in entities and entities["GPE"] not in seen_entities:
            question_list.append(question_templates["historical_significance"].format(topic=entities["GPE"]))
            seen_entities.add(entities["GPE"])

        if "ORG" in entities and entities["ORG"] not in seen_entities:
            question_list.append(f"What is {entities['ORG']} and what role does it play?")
            seen_entities.add(entities["ORG"])

        # 2️⃣ CAUSAL RELATIONSHIP QUESTIONS (Generalized)
        if any(tok.text.lower() in ["because", "due to", "since", "as", "leads to"] for tok in sent):
            question_list.append("What are the causes and effects of this phenomenon?")

        # 3️⃣ EXPLANATORY & CONCEPTUAL QUESTIONS (Filtered & Smart Selection)
        for token in sent:
            if token.pos_ in {"NOUN", "PROPN"} and token.text.lower() not in {"it", "they", "this"}:
                main_concepts.append(token.text)

        if main_concepts:
            topic = " ".join(main_concepts[:2])  # Use only key terms, avoid redundancy
            if topic not in seen_entities:
                selected_templates = random.sample(list(question_templates.values())[:3], min(2, len(question_templates)))  # Pick 2 varied questions
                for template in selected_templates:
                    question_list.append(template.format(topic=topic))
                seen_entities.add(topic)

        # 4️⃣ SPECIAL HANDLING FOR COUNTRIES & PLACES
        if "GPE" in entities and entities["GPE"] not in seen_entities:
            country_questions = [
                question_templates["scientific_contribution"].format(topic=entities["GPE"]),
                question_templates["global_impact"].format(topic=entities["GPE"])
            ]
            question_list.extend(country_questions)
            seen_entities.add(entities["GPE"])

        # 5️⃣ FINAL CLEANUP: Limit number of questions per sentence, shuffle for UX
        question_list = list(set(question_list))[:max_questions]
        random.shuffle(question_list)
        questions.extend(question_list)

    return questions

# Example text
text = """Albert Einstein published a paper in 1915. He was born in Germany. He wrote about Time Dilation, the difference in elapsed time as measured by two clocks, either because of a relative velocity between them (special relativity) or a difference in gravitational potential between their locations (general relativity)."""

# Generate questions
questions = generate_questions(text)

# Print generated questions
for q in questions:
    print(q)
