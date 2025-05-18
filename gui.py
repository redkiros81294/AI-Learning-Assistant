import os
import gradio as gr

from bot.loader import load_pdfs_from_dir
from bot.preprocessor import preprocess_documents
from bot.embedder import Embedder
from bot.trainer import fine_tune_gpt2
from bot.generator import generate_text_answer
from bot.voice import text_to_speech


PDF_DIR = "data"
TEXT_DUMP = "data/combined_notes.txt"
OPENAI_KEY = os.getenv("OPENAI_API_KEY", None)

raw_docs = load_pdfs_from_dir(PDF_DIR)
processed = preprocess_documents(raw_docs)


with open(TEXT_DUMP, "w") as f:
    for doc in processed:
        f.write(doc["text"] + "\n\n")


embedder = Embedder(openai_api_key=OPENAI_KEY)
embedded_docs = embedder.embed_documents(processed)

def qa_with_voice(question: str, speak:bool):
    """
    1) Generate a text answer using your retrieval or direct GPT-2
    2) Optionally generate an audio file
    """

    text = generate_text_answer(question)
    audio = None
    if speak:
        audio = text_to_speech(text, "response.mp3")
    return text, audio

iface = gr.Interface(
    fn=qa_with_voice,
    inputs=[
        gr.Textbox(label="Ask question"),
        gr.Checkbox(label="Speak Answer?")
    ],
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Audio(label="Audio Response")
    ],
    title="AI Learning Assistant ",
)

if __name__ == "__main__":
    iface.launch()