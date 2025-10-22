from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
from preprocess import load_data
import os

def train(train_csv, test_csv):
    X_train, y_train, X_test, y_test = load_data(train_csv, test_csv)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate on test set
    test_preds = pipeline.predict(X_test)
    print("Test Set Evaluation:")
    print(classification_report(y_test, test_preds))

    # Save model
    os.makedirs("../models", exist_ok=True)
    joblib.dump(pipeline, "../models/sentiment_model.pkl")
    print("Model saved at ../models/sentiment_model.pkl")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python train.py train.csv test.csv")
    else:
        train(sys.argv[1], sys.argv[2])
