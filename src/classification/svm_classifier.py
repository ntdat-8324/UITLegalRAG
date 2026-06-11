import pickle
import os
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd

class SVMDocumentClassifier:
    def __init__(self, max_features: int = 5000):
        self.max_features = max_features
        self.model = SVC(kernel='linear', probability=True, random_state=42)
        self.vectorizer = TfidfVectorizer(max_features=self.max_features)
        self.label_encoder = LabelEncoder()
        
    def train(self, df_train: pd.DataFrame):
        """Train SVM classifier on dataframe with 'processed_question' and 'document' columns."""
        X_train = df_train['processed_question']
        y_train = self.label_encoder.fit_transform(df_train['document'])
        
        X_train_encoded = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_encoded, y_train)
        
    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, 'svm_model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
        with open(os.path.join(model_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(os.path.join(model_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
            
    def load(self, model_dir: str):
        with open(os.path.join(model_dir, 'svm_model.pkl'), 'rb') as f:
            self.model = pickle.load(f)
        with open(os.path.join(model_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)
