import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize

# List of required NLTK packages for modern NLTK versions
REQUIRED_NLTK_RESOURCES = [
    'punkt',
    'punkt_tab',
    'averaged_perceptron_tagger',
    'averaged_perceptron_tagger_eng',
    'vader_lexicon'
]

# Ensure required NLTK resources are available locally
for resource in REQUIRED_NLTK_RESOURCES:
    try:
        nltk.download(resource, quiet=True)
    except Exception as e:
        print(f"Warning downloading NLTK resource '{resource}': {e}")


class ReviewFeatureExtractor:
    """
    Extracts 15+ behavioral and linguistic features from raw review text:
    - Text length & word count metrics
    - Lexical diversity & average word length
    - Punctuation & capitalization ratios (exclamations, uppercase words)
    - Part-of-Speech (POS) tag ratios (Nouns, Verbs, Adjectives, Adverbs)
    - VADER Sentiment analysis scores (Positive, Negative, Neutral, Compound)
    """

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def extract_features(self, text: str) -> dict:
        text = str(text) if text is not None else ""
        
        # Tokenization
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        
        word_count = len(words) if len(words) > 0 else 1
        char_count = len(text)
        sentence_count = len(sentences) if len(sentences) > 0 else 1

        # Part-of-Speech (POS) Tagging
        pos_tags = nltk.pos_tag(words)
        noun_count = sum(1 for _, tag in pos_tags if tag.startswith('NN'))
        verb_count = sum(1 for _, tag in pos_tags if tag.startswith('VB'))
        adj_count = sum(1 for _, tag in pos_tags if tag.startswith('JJ'))
        adv_count = sum(1 for _, tag in pos_tags if tag.startswith('RB'))

        # Sentiment Analysis via VADER
        sentiment_scores = self.sia.polarity_scores(text)

        # Behavioral markers: Exclamations & Uppercase usage
        exclamation_count = text.count('!')
        uppercase_words = sum(1 for w in words if w.isupper() and len(w) > 1)

        return {
            # Structural features
            "char_count": char_count,
            "word_count": word_count,
            "avg_word_len": sum(len(w) for w in words) / word_count,
            "sentence_count": sentence_count,
            
            # Linguistic ratios
            "uppercase_ratio": uppercase_words / word_count,
            "exclamation_ratio": exclamation_count / word_count,
            "noun_ratio": noun_count / word_count,
            "verb_ratio": verb_count / word_count,
            "adj_ratio": adj_count / word_count,
            "adv_ratio": adv_count / word_count,
            "lexical_diversity": len(set(words)) / word_count,
            
            # Sentiment features
            "pos_score": sentiment_scores['pos'],
            "neg_score": sentiment_scores['neg'],
            "neu_score": sentiment_scores['neu'],
            "compound_sentiment": sentiment_scores['compound']
        }