# bot/voice.py
import os
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

class VoiceGenerator:
    def __init__(self):
        self.use_coqui = False
        try:
            from TTS.api import TTS
            # Initialize Coqui TTS model
            self.tts = TTS("tts_models/en/vctk/vits")
            # Check if model is multi-speaker: if so, fetch available speakers
            self.coqui_speakers = []
            if hasattr(self.tts, "speakers") and isinstance(self.tts.speakers, list) and self.tts.speakers:
                self.coqui_speakers = self.tts.speakers
                logger.info(f"Coqui model supports multiple speakers: {len(self.coqui_speakers)} available.")
            else:
                # Some models store speakers differently; attempt attribute
                try:
                    spk_attr = getattr(self.tts, "speaker_ids", None)
                    if isinstance(spk_attr, list) and spk_attr:
                        self.coqui_speakers = spk_attr
                        logger.info(f"Coqui model multi-speaker via speaker_ids: {len(self.coqui_speakers)} available.")
                except Exception:
                    pass
            self.use_coqui = True
            logger.info("Using Coqui TTS for voice generation.")
        except Exception as e:
            # Fallback to pyttsx3 if Coqui unavailable
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.lock = threading.Lock()
            logger.info("Coqui TTS not available or failed to load, falling back to pyttsx3.")

    def generate_async(self, text: str, path: str) -> threading.Thread:
        """
        Generate TTS audio asynchronously, saving to `path`.
        If using pyttsx3 fallback, switches .mp3 to .wav automatically.
        Returns the Thread object; caller can join() on it.
        """
        # Determine final path before thread, to avoid UnboundLocalError
        final_path = path
        if not self.use_coqui:
            # Change extension to .wav for pyttsx3
            if final_path.lower().endswith(".mp3"):
                final_path = final_path[:-4] + ".wav"
            else:
                final_path = final_path + ".wav"

        def _generate(p=final_path):
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(p), exist_ok=True)

                if self.use_coqui:
                    # Coqui TTS: handle multi-speaker if needed
                    if self.coqui_speakers:
                        # Pick a default speaker (e.g., the first)
                        speaker = self.coqui_speakers[0]
                        # Some Coqui versions expect parameter name 'speaker' or 'speaker_wav' etc.
                        # The TTS.api.TTS.tts_to_file signature usually accepts speaker=...
                        try:
                            self.tts.tts_to_file(text=text, speaker=speaker, file_path=p)
                        except TypeError:
                            # Fallback if signature differs: try without speaker param
                            logger.warning(f"Failed passing speaker param; retrying without speaker for Coqui.")
                            self.tts.tts_to_file(text=text, file_path=p)
                    else:
                        # Single-speaker model
                        self.tts.tts_to_file(text=text, file_path=p)
                else:
                    # pyttsx3 fallback; thread-safe via lock
                    with self.lock:
                        self.engine.save_to_file(text, p)
                        self.engine.runAndWait()

                # Verify file creation
                if not Path(p).exists():
                    raise FileNotFoundError(f"Audio file {p} was not created")
                logger.info(f"Audio saved to {p}")
            except Exception as e:
                logger.error(f"Voice generation failed for path {p}: {str(e)}")
                # Remove partial file if exists
                try:
                    if Path(p).exists():
                        Path(p).unlink()
                except Exception:
                    pass

        thread = threading.Thread(target=_generate)
        thread.start()
        return thread
