import os
import sys
import urllib.request
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import py_vncorenlp
from src.config.settings import settings

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    urllib.request.urlretrieve(url, dest_path)

def main():
    print(f"Setting up VnCoreNLP at: {settings.vncorenlp_dir}")
    os.makedirs(settings.vncorenlp_dir, exist_ok=True)
    
    try:
        # Manual download because py_vncorenlp uses 'wget' which fails on Windows
        jar_url = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar"
        jar_path = os.path.join(settings.vncorenlp_dir, "VnCoreNLP-1.1.1.jar")
        if not os.path.exists(jar_path):
            download_file(jar_url, jar_path)
            
        # py_vncorenlp also looks for VnCoreNLP-1.2.jar sometimes depending on version, let's get 1.2
        jar_12_url = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.2.jar"
        jar_12_path = os.path.join(settings.vncorenlp_dir, "VnCoreNLP-1.2.jar")
        if not os.path.exists(jar_12_path):
            download_file(jar_12_url, jar_12_path)

        models_dir = os.path.join(settings.vncorenlp_dir, "models", "wordsegmenter")
        os.makedirs(models_dir, exist_ok=True)

        vocab_url = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab"
        rdr_url = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr"
        
        vocab_path = os.path.join(models_dir, "vi-vocab")
        rdr_path = os.path.join(models_dir, "wordsegmenter.rdr")
        
        if not os.path.exists(vocab_path):
            download_file(vocab_url, vocab_path)
        if not os.path.exists(rdr_path):
            download_file(rdr_url, rdr_path)
            
        print("VnCoreNLP downloaded successfully.")
        
        # Test loading
        print("Testing VnCoreNLP loading...")
        abs_dir = os.path.abspath(settings.vncorenlp_dir)
        rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=abs_dir)
        text = "Trường Đại học Công nghệ Thông tin"
        
        # Avoid charmap encode error on Windows
        sys.stdout.reconfigure(encoding='utf-8')
        print(f"Test segment: {rdrsegmenter.word_segment(text)}")
        print("Test successful!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Make sure Java (JDK 8+) is installed and accessible in your system PATH.")

if __name__ == "__main__":
    main()
