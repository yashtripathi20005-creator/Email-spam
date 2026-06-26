import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Create a sample dataset (in practice, you'd load a real dataset like the Enron or SpamAssassin)
# For demonstration, we create a balanced dataset with 1000 samples
def create_sample_data():
    spam_texts = [
        "Congratulations! You've won a free iPhone. Click here to claim now.",
        "Urgent: Your bank account has been compromised. Verify your details immediately.",
        "Get rich quick with this amazing investment opportunity. Act now!",
        "You have been selected for a free vacation. Call this number to book.",
        "Limited time offer: Buy now and get 50% off on all products.",
        "Dear customer, your PayPal account has been limited. Update your info.",
        "Earn $5000 per week working from home. No experience needed.",
        "You are the lucky winner of our lottery. Send us your bank details.",
        "Free pills that will change your life. Order today!",
        "Your credit card has been charged $999. Call us to dispute this."
    ] * 50  # 500 spam samples

    ham_texts = [
        "Hi John, can we meet tomorrow for the project discussion?",
        "Don't forget to bring the documents for the meeting at 3 PM.",
        "Happy birthday! Hope you have a great day with your family.",
        "Please review the attached report and send your feedback.",
        "The conference call has been rescheduled to Friday morning.",
        "Can you pick up some groceries on your way home?",
        "Your appointment with Dr. Smith is confirmed for next Monday.",
        "Let's have lunch together this weekend. What do you think?",
        "The software update is complete. Please restart your computer.",
        "Thank you for your application. We will get back to you soon."
    ] * 50  # 500 ham samples

    # Combine and create labels
    texts = spam_texts + ham_texts
    labels = [1] * len(spam_texts) + [0] * len(ham_texts)
    
    # Shuffle the data
    import random
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    
    return list(texts), list(labels)

def train_model():
    print("Loading dataset...")
    X, y = create_sample_data()
    
    print(f"Dataset size: {len(X)} samples")
    print(f"Spam: {sum(y)} samples")
    print(f"Ham: {len(y) - sum(y)} samples")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create a pipeline with TF-IDF and Naive Bayes
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )),
        ('classifier', MultinomialNB(alpha=0.1))
    ])
    
    print("Training the model...")
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the model
    model_dir = 'model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_path = os.path.join(model_dir, 'spam_classifier_model.joblib')
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to {model_path}")
    
    return pipeline

if __name__ == "__main__":
    train_model()
