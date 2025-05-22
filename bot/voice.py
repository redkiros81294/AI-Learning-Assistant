import os
import logging
import pyttsx3
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

class VoiceGenerator:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.lock = threading.Lock()
    
    # def generate_async(self, text: str, path: str) -> threading.Thread:
    #     def _generate():
    #         try:
    #             os.makedirs(os.path.dirname(path), exist_ok=True)
    #             with self.lock:
    #                 if Path(path).exists():
    #                     Path(path).unlink()
    #                 self.engine.save_to_file(text, path)
    #                 self.engine.runAndWait()
    #                 logger.info(f"Audio saved to {path}")
    #         except Exception as e:
    #             logger.error(f"Voice generation failed: {str(e)}")

    #     thread = threading.Thread(target=_generate)
    #     thread.start()
    #     return thread
    def generate_async(self, text: str, path: str) -> threading.Thread:
        def _generate():
            try:
                self.engine.save_to_file(text, path)
                self.engine.runAndWait()
                if not Path(path).exists():
                    raise FileNotFoundError(f"Audio file {path} not created")
                logger.info(f"Audio saved to {path}")
            except Exception as e:
                logger.error(f"Voice failed: {str(e)}")
                if Path(path).exists():
                    Path(path).unlink()
        thread = threading.Thread(target=_generate)
        thread.start()
        return thread