import numpy as np
import string
import re
import textstat
import os
import nltk
from pathlib import Path

current_dir = Path(__file__).parent
nltk_data_path = current_dir.parent / "notebooks" / "nltk_data"

nltk.data.path += [
    str(nltk_data_path),  # Local development
    "/root/nltk_data",    # Docker container
    nltk.data.path[0]     # Default system path
]

try:
    nltk.data.find('corpora/cmudict')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading required NLTK datasets...")
    nltk.download('cmudict', download_dir=str(nltk_data_path))
    nltk.download('punkt', download_dir=str(nltk_data_path))
    nltk.download('stopwords', download_dir=str(nltk_data_path))
    nltk.download('punkt_tab', download_dir=str(nltk_data_path))
    nltk.download('wordnet', download_dir=str(nltk_data_path))

from nltk.corpus import cmudict, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from textblob import TextBlob
from gensim.models import LsiModel
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel

cmu_dict = cmudict.dict()

### Define custom functions


def word_count(text):
    if not isinstance(text, str):
       text = str(text)
    return len(text.split())


def basic_cleaning(text):
    if not isinstance(text, str):  # Convert to string if it's not
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


def cons_density(text):

    consonnant = sum(1 for char in text if char.isalpha() and char not in "aeiou")
    vowel = sum(1 for char in text if char.isalpha() and char in "aeiou")
    total_letters = vowel + consonnant
    return round((consonnant/(vowel + consonnant)),3) if total_letters > 0 else 0


def get_word_stress(word):
    if word in cmu_dict:
        return sum(int(char) for syllable in cmu_dict[word][0] for char in syllable if char.isdigit())
    return 0

def get_sentence_stress(sentence):
    words = sentence.split()
    stress_values = [get_word_stress(word) for word in words]
    return sum(stress_values)


def redundance(text):
    # give a redundance score, considering the lenght of each text, if a lemmatized words appears more than three times the mean, it is considered redundant.

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    clean_tokens = [w for w in tokens if w not in stop_words]

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(w) for w in clean_tokens]

    word_counts = Counter(lemmatized_tokens)
    mean_freq = sum(word_counts.values()) / len(word_counts) if len(word_counts)!= 0 else 0

    if mean_freq != 0:
        score = sum(1 for word, count in word_counts.items() if count > 2.5 * mean_freq)
    else:
        score = 0

    return score


def sentiment_polarity(text):
    sent_pol = TextBlob(text).sentiment.polarity
    return abs(round(sent_pol,3))


def word_choice(text):
    common_ai_words =["commendable",'transhumanist', 'meticulous', 'elevate','hello', 'tapestry','leverage',
                  'journey', 'headache','resonate','testament','explore', 'binary','delve',
                  'enrich', 'seamless','multifaceted', 'sorry','foster', 'convey', 'beacon',
                  'interplay', 'oh', 'navigate','form','adhere','cannot', 'landscape','remember',
                  'paramount', 'comprehensive', 'placeholder','grammar','real','summary','symphony',
                  'furthermore','relationship','ultimately','profound','art','supercharge','evolve',
                  'beyoud','reimagine','vibrant', 'robust','pivotal','certainly','quinoa','orchestrate','align',
                  'diverse','recommend','annals','note','employ','bustling','indeed','digital','enigma', 'outfit',
                  'indelible','refrain','culture','treat','emerge','meticulous','esteemed','weight','whimsical','bespoke',
                  'highlight','antagonist','unlock','key','breakdown','tailor','misinformation','treasure','paradigm','captivate',
                  'song','underscore','calculate','especially','climate','hedging','inclusive','exercise','ai','embrace',
                  'level','nuance','career','dynamic','accent','ethos','cheap','firstly','online','goodbye'
                  ]
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    word_count = 0
    for word in text.split():
        if word in common_ai_words:
            word_count += 1

    return word_count


def coherence(text):
    # uses gensim to measure coherence, use the lsi model(latent semantic indexing, coherence c_v because we provide the text)
    tokens = word_tokenize(text)
    if not tokens:
        coherence_score = 0
    else:
        dictionary = corpora.Dictionary([tokens])
        corpus_gensim = [dictionary.doc2bow(tokens)]
        lsa_model = LsiModel(corpus_gensim, id2word=dictionary)

        coherence_model = CoherenceModel(
            model=lsa_model,
            texts=[tokens],
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
    return coherence_score



def reading_ease(text):
    reading_ease= textstat.flesch_reading_ease(text)
    return reading_ease


def gunning_fog(text):
    gunning_fog = textstat.gunning_fog(text)
    return gunning_fog
