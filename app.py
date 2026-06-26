import sys
from spam_classifier import SpamClassifier

def main():
    """
    A simple command-line interface for the spam classifier.
    """
    print("=" * 60)
    print("EMAIL SPAM CLASSIFIER")
    print("=" * 60)
    
    try:
        classifier = SpamClassifier()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python train_model.py' first to train the model.")
        return
    
    print("\nEnter your email text (type 'quit' to exit):")
    print("-" * 60)
    
    while True:
        print("\n> ", end="")
        text = input().strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not text:
            print("Please enter some text.")
            continue
        
        try:
            pred, label, confidence = classifier.predict(text)
            print(f"\nResult: {label}")
            print(f"Confidence: {confidence:.2f}%")
            
            # Show feature importance if available
            if confidence > 80:
                print("The model is very confident in this prediction.")
            
        except Exception as e:
            print(f"Error making prediction: {e}")

if __name__ == "__main__":
    main()
