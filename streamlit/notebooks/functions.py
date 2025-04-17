import spacy
import re
import string
import textstat
from collections import Counter


nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def basic_cleaning(text):
    if not isinstance(text, str):
       text = str(text)
    # Remove whitespace
    prepoc_text = text.strip()
    # Lowercasing
    prepoc_text = prepoc_text.lower()
    # remove digits
    prepoc_text = "".join(char for char in prepoc_text if not char.isdigit())
    # remove punctuation
    for punctuation in string.punctuation:
        prepoc_text = prepoc_text.replace(punctuation," ")
    # remove regex
    prepoc_text = re.sub('<[^<]+?',"",prepoc_text)

    return prepoc_text

def spacy_preprocessor(text):
    # Advanced preprocessing with spaCy (to be used for TF-IDF only)
    if not isinstance(text, str):
        text = str(text)
    
    # Apply basic cleaning first
    text = basic_cleaning(text)
    
    # Process with spaCy
    doc = nlp(text)
    
    # Lemmatization + stopword removal + token filtering
    tokens = [
        token.lemma_.lower().strip() 
        for token in doc 
        if not token.is_stop 
        and not token.is_punct 
        and len(token.lemma_) > 2
    ]
    
    return " ".join(tokens)


def redundance(text):
    # give a redundance score, considering the lenght of each text, if a lemmatized words appears more than three times the mean, it is considered redundant.

    doc = nlp(text)
    lemmatized_tokens = [
        token.lemma_.lower().strip() 
        for token in doc 
        if not token.is_stop 
        and not token.is_punct 
        and len(token.lemma_) > 2
    ]

    if not lemmatized_tokens:
        return 0.0
    
    if len(lemmatized_tokens) < 10:
        return 0.0 
    
    total_words = len(lemmatized_tokens)
    word_counts = Counter(lemmatized_tokens)
    
    # Calculate threshold: Flag words appearing >15% of the time
    threshold = 0.15 * total_words
    redundant_words = sum(1 for count in word_counts.values() if count > threshold)
    
    # Normalized score (0 to 1)
    return round(redundant_words / total_words, 2)
    
    

def reading_ease(text):
    reading_ease= textstat.flesch_reading_ease(text)
    return reading_ease


def gunning_fog(text):
    gunning_fog = textstat.gunning_fog(text)
    return gunning_fog
