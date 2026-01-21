"""
Merchant Feedback Sentiment Analysis

NLP module for analyzing merchant support tickets and feedback.
Uses both rule-based and ML approaches for sentiment classification.

Aligns with JD: "Natural language processing" skill requirement
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SentimentResult:
    """Container for sentiment analysis results."""
    text: str
    sentiment: str
    confidence: float
    keywords: List[str]
    category: Optional[str] = None


@dataclass
class SentimentMetrics:
    """Container for model evaluation metrics."""
    accuracy: float
    report: str
    class_distribution: Dict[str, int]


class SentimentAnalyzer:
    """
    Multi-approach sentiment analyzer for merchant feedback.
    
    Approaches:
    1. Rule-based (lexicon) for quick classification
    2. ML-based (TF-IDF + classifier) for nuanced analysis
    
    Categories handled:
    - Pricing/fees
    - Technical/integration
    - Support quality
    - Features/functionality
    - General feedback
    """
    
    # Sentiment lexicons
    POSITIVE_WORDS = {
        'great', 'excellent', 'amazing', 'fantastic', 'wonderful', 'love',
        'helpful', 'easy', 'fast', 'smooth', 'reliable', 'satisfied',
        'recommend', 'improved', 'efficient', 'professional', 'responsive',
        'seamless', 'perfect', 'outstanding', 'superb', 'impressed'
    }
    
    NEGATIVE_WORDS = {
        'terrible', 'awful', 'horrible', 'poor', 'bad', 'slow', 'difficult',
        'frustrated', 'disappointed', 'issue', 'problem', 'error', 'bug',
        'expensive', 'confusing', 'unreliable', 'delayed', 'held', 'frozen',
        'complicated', 'unresponsive', 'failed', 'lost', 'unhelpful'
    }
    
    # Category keywords
    CATEGORY_KEYWORDS = {
        'billing': ['fee', 'rate', 'charge', 'price', 'cost', 'expensive', 'billing', 'invoice'],
        'technical': ['api', 'integration', 'webhook', 'error', 'bug', 'code', 'sdk', 'technical'],
        'support': ['support', 'help', 'response', 'wait', 'ticket', 'agent', 'service'],
        'features': ['feature', 'dashboard', 'report', 'analytics', 'tool', 'function'],
        'disputes': ['dispute', 'chargeback', 'refund', 'fraud', 'held', 'frozen', 'review'],
        'account': ['account', 'verification', 'document', 'limit', 'access']
    }
    
    def __init__(self):
        """Initialize sentiment analyzer."""
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2
        )
        self.classifier = None
        self.is_fitted = False
        
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for analysis.
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned and normalized text
        """
        # Lowercase
        text = text.lower()
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^a-z0-9\s\.\!\?]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """
        Extract key terms from text.
        
        Args:
            text: Preprocessed text
            top_n: Number of keywords to extract
            
        Returns:
            List of important keywords
        """
        words = text.split()
        
        # Find sentiment-related words
        sentiment_words = []
        for word in words:
            if word in self.POSITIVE_WORDS or word in self.NEGATIVE_WORDS:
                sentiment_words.append(word)
        
        # Find category words
        category_words = []
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            for word in words:
                if word in keywords:
                    category_words.append(word)
        
        # Combine and deduplicate
        all_keywords = sentiment_words + category_words
        return list(dict.fromkeys(all_keywords))[:top_n]
    
    def _detect_category(self, text: str) -> str:
        """
        Detect feedback category from text.
        
        Args:
            text: Preprocessed text
            
        Returns:
            Detected category
        """
        words = set(text.split())
        
        category_scores = {}
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            score = len(words.intersection(keywords))
            category_scores[category] = score
        
        if max(category_scores.values()) == 0:
            return 'general'
        
        return max(category_scores, key=category_scores.get)
    
    def analyze_rule_based(self, text: str) -> SentimentResult:
        """
        Rule-based sentiment analysis using lexicons.
        
        Args:
            text: Text to analyze
            
        Returns:
            SentimentResult with classification
        """
        processed = self._preprocess_text(text)
        words = set(processed.split())
        
        # Count sentiment words
        positive_count = len(words.intersection(self.POSITIVE_WORDS))
        negative_count = len(words.intersection(self.NEGATIVE_WORDS))
        
        total = positive_count + negative_count
        if total == 0:
            sentiment = 'neutral'
            confidence = 0.5
        elif positive_count > negative_count:
            sentiment = 'positive'
            confidence = positive_count / total
        elif negative_count > positive_count:
            sentiment = 'negative'
            confidence = negative_count / total
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        return SentimentResult(
            text=text,
            sentiment=sentiment,
            confidence=round(confidence, 3),
            keywords=self._extract_keywords(processed),
            category=self._detect_category(processed)
        )
    
    def fit(self, texts: List[str], labels: List[str]) -> SentimentMetrics:
        """
        Train ML-based sentiment classifier.
        
        Args:
            texts: List of text samples
            labels: List of sentiment labels
            
        Returns:
            SentimentMetrics with model performance
        """
        # Preprocess
        processed_texts = [self._preprocess_text(t) for t in texts]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Vectorize
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train classifier
        self.classifier = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        self.classifier.fit(X_train_vec, y_train)
        self.is_fitted = True
        
        # Evaluate
        y_pred = self.classifier.predict(X_test_vec)
        
        return SentimentMetrics(
            accuracy=accuracy_score(y_test, y_pred),
            report=classification_report(y_test, y_pred),
            class_distribution=dict(Counter(labels))
        )
    
    def analyze_ml(self, text: str) -> SentimentResult:
        """
        ML-based sentiment analysis.
        
        Args:
            text: Text to analyze
            
        Returns:
            SentimentResult with classification
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first or use analyze_rule_based()")
        
        processed = self._preprocess_text(text)
        
        # Vectorize and predict
        text_vec = self.vectorizer.transform([processed])
        prediction = self.classifier.predict(text_vec)[0]
        probabilities = self.classifier.predict_proba(text_vec)[0]
        confidence = max(probabilities)
        
        return SentimentResult(
            text=text,
            sentiment=prediction,
            confidence=round(confidence, 3),
            keywords=self._extract_keywords(processed),
            category=self._detect_category(processed)
        )
    
    def analyze(self, text: str, method: str = 'hybrid') -> SentimentResult:
        """
        Analyze sentiment using specified method.
        
        Args:
            text: Text to analyze
            method: 'rule', 'ml', or 'hybrid'
            
        Returns:
            SentimentResult with classification
        """
        if method == 'rule':
            return self.analyze_rule_based(text)
        elif method == 'ml':
            return self.analyze_ml(text)
        elif method == 'hybrid':
            # Use both and combine
            rule_result = self.analyze_rule_based(text)
            
            if self.is_fitted:
                ml_result = self.analyze_ml(text)
                
                # Use ML if confident, otherwise rule-based
                if ml_result.confidence > 0.7:
                    return ml_result
                elif rule_result.confidence > 0.6:
                    return rule_result
                else:
                    # Average confidence, prefer ML prediction
                    return SentimentResult(
                        text=text,
                        sentiment=ml_result.sentiment,
                        confidence=(ml_result.confidence + rule_result.confidence) / 2,
                        keywords=rule_result.keywords,
                        category=rule_result.category
                    )
            
            return rule_result
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def batch_analyze(
        self, 
        texts: List[str], 
        method: str = 'hybrid'
    ) -> pd.DataFrame:
        """
        Analyze multiple texts.
        
        Args:
            texts: List of texts to analyze
            method: Analysis method
            
        Returns:
            DataFrame with analysis results
        """
        results = []
        for text in texts:
            result = self.analyze(text, method)
            results.append({
                'text': result.text[:100] + '...' if len(result.text) > 100 else result.text,
                'sentiment': result.sentiment,
                'confidence': result.confidence,
                'category': result.category,
                'keywords': ', '.join(result.keywords)
            })
        
        return pd.DataFrame(results)


class FeedbackCategorizer:
    """
    Categorize merchant feedback for routing and analysis.
    
    Uses keyword matching and optional ML classification
    to route tickets to appropriate teams.
    """
    
    ROUTING_RULES = {
        'billing': {
            'team': 'Finance',
            'priority_keywords': ['urgent', 'overcharged', 'refund'],
            'sla_hours': 24
        },
        'technical': {
            'team': 'Engineering',
            'priority_keywords': ['down', 'error', 'broken', 'crash'],
            'sla_hours': 4
        },
        'support': {
            'team': 'Customer Success',
            'priority_keywords': ['escalate', 'manager', 'complaint'],
            'sla_hours': 8
        },
        'disputes': {
            'team': 'Risk',
            'priority_keywords': ['fraud', 'chargeback', 'frozen'],
            'sla_hours': 4
        },
        'account': {
            'team': 'Account Management',
            'priority_keywords': ['suspended', 'locked', 'verify'],
            'sla_hours': 12
        },
        'features': {
            'team': 'Product',
            'priority_keywords': ['request', 'suggestion', 'need'],
            'sla_hours': 48
        },
        'general': {
            'team': 'Customer Success',
            'priority_keywords': [],
            'sla_hours': 24
        }
    }
    
    def __init__(self, sentiment_analyzer: SentimentAnalyzer):
        """
        Initialize categorizer.
        
        Args:
            sentiment_analyzer: Configured SentimentAnalyzer instance
        """
        self.analyzer = sentiment_analyzer
    
    def categorize(self, text: str) -> Dict:
        """
        Categorize feedback and determine routing.
        
        Args:
            text: Feedback text
            
        Returns:
            Dictionary with category, routing, and priority info
        """
        result = self.analyzer.analyze(text)
        category = result.category
        routing = self.ROUTING_RULES.get(category, self.ROUTING_RULES['general'])
        
        # Determine priority
        text_lower = text.lower()
        is_urgent = any(kw in text_lower for kw in routing['priority_keywords'])
        is_negative = result.sentiment == 'negative'
        
        if is_urgent and is_negative:
            priority = 'critical'
            sla_hours = routing['sla_hours'] // 2
        elif is_urgent or is_negative:
            priority = 'high'
            sla_hours = routing['sla_hours']
        else:
            priority = 'normal'
            sla_hours = routing['sla_hours'] * 2
        
        return {
            'category': category,
            'sentiment': result.sentiment,
            'confidence': result.confidence,
            'team': routing['team'],
            'priority': priority,
            'sla_hours': sla_hours,
            'keywords': result.keywords
        }
    
    def batch_categorize(self, feedback_df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Categorize batch of feedback.
        
        Args:
            feedback_df: DataFrame with feedback
            text_column: Column containing text
            
        Returns:
            DataFrame with categorization results
        """
        categorizations = []
        
        for _, row in feedback_df.iterrows():
            cat = self.categorize(row[text_column])
            categorizations.append(cat)
        
        cat_df = pd.DataFrame(categorizations)
        return pd.concat([feedback_df.reset_index(drop=True), cat_df], axis=1)


def main():
    """Demo: Sentiment analysis on merchant feedback."""
    import sys
    sys.path.insert(0, '../data')
    from generator import PayPalDataGenerator, DataConfig
    
    # Generate sample data
    config = DataConfig()
    generator = PayPalDataGenerator(config)
    feedback_df = generator.generate_merchant_feedback(num_feedbacks=1000)
    
    print("=" * 60)
    print("SENTIMENT ANALYSIS MODEL")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Train ML model
    texts = feedback_df['text'].tolist()
    labels = feedback_df['sentiment_label'].tolist()
    
    metrics = analyzer.fit(texts, labels)
    
    print(f"\nML Model Performance:")
    print(f"  Accuracy: {metrics.accuracy:.4f}")
    print(f"\nClassification Report:")
    print(metrics.report)
    
    # Test individual analysis
    print("\n" + "=" * 60)
    print("SAMPLE ANALYSIS")
    print("=" * 60)
    
    test_texts = [
        "The checkout process is amazing! Our conversion rate improved by 20%.",
        "Transaction fees are way too high. We're considering switching to a competitor.",
        "Looking for documentation on webhook implementation for order notifications.",
        "Funds have been held for 3 weeks without any explanation. Very frustrated!",
        "The new dashboard features are excellent for tracking business metrics."
    ]
    
    for text in test_texts:
        result = analyzer.analyze(text, method='hybrid')
        print(f"\nText: {text[:60]}...")
        print(f"  Sentiment: {result.sentiment} ({result.confidence:.2%} confidence)")
        print(f"  Category: {result.category}")
        print(f"  Keywords: {', '.join(result.keywords)}")
    
    # Batch categorization
    print("\n" + "=" * 60)
    print("FEEDBACK CATEGORIZATION")
    print("=" * 60)
    
    categorizer = FeedbackCategorizer(analyzer)
    categorized_df = categorizer.batch_categorize(feedback_df.head(100))
    
    print("\nCategory Distribution:")
    print(categorized_df['category'].value_counts())
    
    print("\nPriority Distribution:")
    print(categorized_df['priority'].value_counts())
    
    print("\nRouting Distribution:")
    print(categorized_df['team'].value_counts())
    
    # Sentiment by category
    print("\nSentiment by Category:")
    print(categorized_df.groupby(['category', 'sentiment']).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
