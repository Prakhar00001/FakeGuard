import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required NLTK resources silently
for resource in ['punkt', 'averaged_perceptron_tagger', 'vader_lexicon']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'help/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

class ReviewFeatureExtractor:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def extract_features(self, text: str) -> dict:
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        word_count = len(words) if len(words) > 0 else 1
        
        # POS Tagging
        pos_tags = nltk.pos_tag(words)
        noun_count = sum(1 for _, tag in pos_tags if tag.startswith('NN'))
        verb_count = sum(1 for _, tag in pos_tags if tag.startswith('VB'))
        adj_count = sum(1 for _, tag in pos_tags if tag.startswith('JJ'))
        adv_count = sum(1 for _, tag in pos_tags if tag.startswith('RB'))
        
        # Sentiment
        sentiment_scores = self.sia.polarity_scores(text)
        
        # Exclamation & Punctuation metrics
        exclamation_count = text.count('!')
        uppercase_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        
        return {
            "char_count": len(text),
            "word_count": word_count,
            "avg_word_len": sum(len(w) for w in words) / word_count,
            "sentence_count": len(sentences),
            "uppercase_ratio": uppercase_words / word_count,
            "exclamation_ratio": exclamation_count / word_count,
            "noun_ratio": noun_count / word_count,
            "verb_ratio": verb_count / word_count,
            "adj_ratio": adj_count / word_count,
            "adv_ratio": adv_count / word_count,
            "lexical_diversity": len(set(words)) / word_count,
            "pos_score": sentiment_scores['pos'],
            "neg_score": sentiment_scores['neg'],
            "neu_score": sentiment_scores['neu'],
            "compound_sentiment": sentiment_scores['compound']
        }