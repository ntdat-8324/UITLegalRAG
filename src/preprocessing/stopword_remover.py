import os
from typing import List

class StopwordRemover:
    """Class to remove Vietnamese stopwords."""
    
    def __init__(self, stopwords_path: str):
        self.stopwords = set()
        if os.path.exists(stopwords_path):
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                self.stopwords = set(f.read().splitlines())
        else:
            print(f"Warning: Stopwords file not found at {stopwords_path}. Stopword removal will be a no-op.")
            
    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords from a space-separated string."""
        if not self.stopwords:
            return text
            
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stopwords]
        return ' '.join(filtered_words)

    def batch_remove(self, texts: List[str]) -> List[str]:
        return [self.remove_stopwords(t) for t in texts]
