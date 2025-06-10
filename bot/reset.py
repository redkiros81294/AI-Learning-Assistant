# bot/reset.py
import shutil
from pathlib import Path

def reset_system():
    folders = [
        "data/embeddings",
        "data/chunk_cache"
    ]
    
    for folder in folders:
        if Path(folder).exists():
            shutil.rmtree(folder)
            print(f"🧹 Cleared {folder}")
        Path(folder).mkdir(parents=True, exist_ok=True)
    
    print("✅ System reset complete")

if __name__ == "__main__":
    reset_system()