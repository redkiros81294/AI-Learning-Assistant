from dia.model import Dia

tts_model = Dia.from_pretrained(
    "nari-labs/Dia-1.6B",
    compute_dtype="float16"
)

def text_to_speech(text: str, output_path: str):
    """Generate TTS audio from text and save it to 'output_path'."""
    # wrap in [S1]/[S2] tags for single speaker prompt
    prompt = f"[S1]{text}"
    audio = tts_model.generate(prompt, use_torch_compile=True)
    tts_model.save_audio(output_path, audio)
    return output_path