import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer

class SpamClassifier:
    def __init__(self, model_path='model/spam_classifier_model.joblib'):
        """
        Initialize the spam classifier with a trained model.
        
        Args:
            model_path (str): Path to the saved model file
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """
        Load the trained model from disk.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}. Please run train_model.py first.")
        
        self.model = joblib.load(self.model_path)
        print("Model loaded successfully!")
    
    def predict(self, text):
        """
        Predict whether a single email is spam or ham.
        
        Args:
            text (str): The email text to classify
            
        Returns:
            tuple: (prediction_label, probability, confidence)
        """
        if not self.model:
            raise ValueError("Model not loaded. Please load the model first.")
        
        # Get prediction and probability
        prediction = self.model.predict([text])[0]
        probabilities = self.model.predict_proba([text])[0]
        
        # For binary classification: probabilities[0] = ham, probabilities[1] = spam
        confidence = max(probabilities) * 100
        label = "Spam" if prediction == 1 else "Ham"
        
        return prediction, label, confidence
    
    def predict_batch(self, texts):
        """
        Predict whether multiple emails are spam or ham.
        
        Args:
            texts (list): List of email texts to classify
            
        Returns:
            list: List of tuples (prediction_label, label, confidence)
        """
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results
    
    def get_feature_importance(self, top_n=10):
        """
        Get the most important features (words) for classification.
        
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            dict: Top features and their importance scores
        """
        if not self.model:
            raise ValueError("Model not loaded.")
        
        # Extract the vectorizer and classifier from the pipeline
        vectorizer = self.model.named_steps['tfidf']
        classifier = self.model.named_steps['classifier']
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Get coefficients (for Naive Bayes, we use log probabilities)
        # For MultinomialNB, we can use feature_log_prob_
        spam_log_prob = classifier.feature_log_prob_[1]  # Spam class
        ham_log_prob = classifier.feature_log_prob_[0]   # Ham class
        
        # Calculate difference (spam - ham) to get importance
        importance = spam_log_prob - ham_log_prob
        
        # Get top features for spam
        top_spam_indices = importance.argsort()[-top_n:][::-1]
        top_spam_features = [(feature_names[i], importance[i]) for i in top_spam_indices]
        
        # Get top features for ham
        top_ham_indices = importance.argsort()[:top_n]
        top_ham_features = [(feature_names[i], importance[i]) for i in top_ham_indices]
        
        return {
            'top_spam_features': top_spam_features,
            'top_ham_features': top_ham_features
        }

# Example usage
if __name__ == "__main__":
    # Test the classifier
    classifier = SpamClassifier()
    
    # Test emails
    test_emails = [
        "Congratulations! You've won a $1000 gift card. Click here to claim now!",
        "Hi, can we schedule a meeting for next week to discuss the project progress?",
        "URGENT: Your account has been hacked. Verify your identity immediately.",
        "Don't forget to bring the presentation slides for tomorrow's meeting."
    ]
    
    print("\nTesting the classifier with sample emails:")
    print("-" * 60)
    
    for email in test_emails:
        pred, label, confidence = classifier.predict(email)
        print(f"Email: {email[:50]}...")
        print(f"Prediction: {label} (Confidence: {confidence:.2f}%)")
        print("-" * 60)
