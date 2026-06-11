import numpy as np
from typing import List

class DocumentPredictor:
    def __init__(self, model, vectorizer, label_encoder, preprocessor=None, stopword_remover=None):
        self.model = model
        self.vectorizer = vectorizer
        self.label_encoder = label_encoder
        self.preprocessor = preprocessor
        self.stopword_remover = stopword_remover
        
    def predict_top_k(self, input_text: str, k: int = 15) -> List[str]:
        # Preprocess text just like in training
        text = input_text.lower()
        if self.preprocessor:
            text = self.preprocessor.preprocess(text)
        if self.stopword_remover:
            text = self.stopword_remover.remove_stopwords(text)
            
        encoded_text = self.vectorizer.transform([text])
        probabilities = self.model.predict_proba(encoded_text)[0]
        
        # Get top k indices
        top_k_indices = np.argsort(probabilities)[-k:][::-1]
        top_k_labels = self.label_encoder.inverse_transform(top_k_indices)
        
        return top_k_labels.tolist()
