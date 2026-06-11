import pandas as pd
from typing import Tuple

def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    """Load CSV data, deduplicate, and perform initial prep."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return pd.DataFrame()
        
    # Standard columns
    if not all(col in df.columns for col in ['context', 'document', 'article']):
        raise ValueError("CSV must contain 'context', 'document', and 'article' columns.")
        
    df = df[['context', 'document', 'article']]
    df = df.drop_duplicates()
    
    # Lowercase context
    df['context'] = df['context'].str.lower()
    
    # Reset index to make sure it's sequential
    df.reset_index(drop=True, inplace=True)
    return df
