import pandas as pd
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path, encoding='latin1').dropna(subset=['text','sentiment'])
    test_df = pd.read_csv(test_path, encoding='latin1').dropna(subset=['text','sentiment'])
    
    train_df['text'] = train_df['text'].apply(clean_text)
    test_df['text'] = test_df['text'].apply(clean_text)
    
    X_train, y_train = train_df['text'].values, train_df['sentiment'].values
    X_test, y_test = test_df['text'].values, test_df['sentiment'].values
    
    return X_train, y_train, X_test, y_test

