import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.config.settings import settings
from src.preprocessing.vncore_segmenter import VNPreprocessor
from src.preprocessing.stopword_remover import StopwordRemover
from src.classification.svm_classifier import SVMDocumentClassifier

def main():
    print("Starting SVM Classifier training pipeline...")
    
    # Check data
    train_path = os.path.join(settings.data_dir, 'train.csv')
    if not os.path.exists(train_path):
        print(f"Error: Training data not found at {train_path}")
        return
        
    print(f"Loading data from {train_path}...")
    df_train = pd.read_csv(train_path)
    
    print("Initializing preprocessors...")
    preprocessor = VNPreprocessor(settings.vncorenlp_dir)
    stopword_remover = StopwordRemover(settings.stopwords_path)
    
    print("Preprocessing questions...")
    # Lowercase -> word segment -> remove stopwords
    df_train['processed_question'] = df_train['question'].str.lower()
    df_train['processed_question'] = df_train['processed_question'].apply(preprocessor.preprocess)
    df_train['processed_question'] = df_train['processed_question'].apply(stopword_remover.remove_stopwords)
    
    print("Training SVM Model...")
    classifier = SVMDocumentClassifier(max_features=5000)
    classifier.train(df_train)
    
    print(f"Saving model to {settings.model_dir}...")
    classifier.save(settings.model_dir)
    
    print("Training complete!")

if __name__ == "__main__":
    main()
