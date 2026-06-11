import py_vncorenlp
from typing import List
import os

class VNPreprocessor:
    """Wrapper class for VnCoreNLP to perform Vietnamese word segmentation."""
    
    def __init__(self, vncorenlp_dir: str):
        self.vncorenlp_dir = vncorenlp_dir
        self.nlp = None

    def _initialize(self):
        """Lazy initialization of the VnCoreNLP model."""
        if self.nlp is None:
            actual_dir = os.path.expanduser(self.vncorenlp_dir)
            if not os.path.exists(os.path.join(actual_dir, "VnCoreNLP-1.1.1.jar")) and not os.path.exists(os.path.join(actual_dir, "VnCoreNLP-1.2.jar")):
                raise FileNotFoundError(
                    f"VnCoreNLP not found in {actual_dir}. "
                    "Please run scripts/setup_vncorenlp.py first."
                )
            
            # py_vncorenlp evil behavior: it changes the CWD! We must restore it.
            original_cwd = os.getcwd()
            try:
                abs_dir = os.path.abspath(actual_dir)
                self.nlp = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=abs_dir)
            finally:
                os.chdir(original_cwd)

    def preprocess(self, text: str) -> str:
        """
        Segment a text string.
        Output is space-separated segmented words (e.g., 'Sinh_viên trường đại_học').
        """
        self._initialize()
        
        # word_segment might return a list of sentences (which are strings) or a list of list of tokens
        # Depending on py_vncorenlp version. We'll join sentences.
        segmented_sentences = self.nlp.word_segment(text)
        
        # segmented_sentences might be a list of strings like ["Ông Nguyễn_Khắc_Chúc đang làm_việc ."]
        if isinstance(segmented_sentences, list):
            # If it's a list of lists of strings
            if len(segmented_sentences) > 0 and isinstance(segmented_sentences[0], list):
                flattened = []
                for sentence in segmented_sentences:
                    flattened.extend(sentence)
                return ' '.join(flattened)
            # If it's a list of strings
            return ' '.join(segmented_sentences)
            
        return str(segmented_sentences)

    def batch_preprocess(self, texts: List[str]) -> List[str]:
        """Process a batch of texts."""
        return [self.preprocess(text) for text in texts]
