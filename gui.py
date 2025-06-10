# gui.py (main application)
import os
import gradio as gr
from bot.loader import load_documents_from_dir
from bot.embedder import Embedder
from bot.generator import AnswerGenerator
from bot.voice import VoiceGenerator
from bot.pdf_qa_module import PDFQASystem
import logging
import gc
import torch
import hashlib
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
_EMBEDDER = None
_GENERATOR = AnswerGenerator()
_VOICE = VoiceGenerator()
_PDF_QA = PDFQASystem()

PDF_CONTEXT = _PDF_QA.load_pdfs_from_directory("data/pdfs")

def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Embedder()
    return _EMBEDDER

def process_documents():
    emb = get_embedder()
    if emb.index.ntotal > 0:
        return "✅ Documents already processed"
    
    try:
        docs = list(load_documents_from_dir("data/texts")) + list(load_documents_from_dir("data/pdfs"))
        if not docs:
            return "❌ No documents found in data/texts"
        
        for doc_id, doc in enumerate(docs):
            emb.embed_and_index(doc_id, doc["text"])
        
        return f"✅ Processed {len(docs)} documents with {emb.index.ntotal} chunks"
    except Exception as e:
        return f"❌ Failed to load documents: {str(e)}"


def handle_query(question, use_voice, get_help):
    emb = get_embedder()
    base_answer = "No relevant information found"
    help_answer = ""
    audio = None

    try:
        # Query embedding and answer generation (unchanged)...
        hits = emb.query(question, top_k=3)
        contexts = []
        for doc_id, chunk_id in hits:
            contexts.append(emb.get_chunk_text(doc_id, chunk_id))
        if contexts:
            combined_context = "\n\n".join(contexts)
            base_answer = _GENERATOR.extract_answer(combined_context, question)
        else:
            base_answer = "No relevant information found in documents."

        # Voice generation
        if use_voice and base_answer and "no relevant" not in base_answer.lower():
            hashed_name = hashlib.md5(base_answer.encode()).hexdigest()
            requested_path = f"data/audio/{hashed_name}.mp3"
            thread = _VOICE.generate_async(base_answer, path=requested_path)
            thread.join(timeout=45)
            mp3_path = requested_path
            wav_path = requested_path[:-4] + ".wav"
            if os.path.exists(mp3_path):
                audio = os.path.abspath(mp3_path)
                logger.info(f"Returning Coqui-generated audio: {audio}")
            elif os.path.exists(wav_path):
                audio = os.path.abspath(wav_path)
                logger.info(f"Returning fallback-generated audio: {audio}")
            else:
                logger.warning(f"Audio file not found after generation: tried {mp3_path} and {wav_path}")

        # External help (Gemini)
        if get_help:
            if PDF_CONTEXT:
                help_answer = _PDF_QA.get_explained_answer(question, PDF_CONTEXT)
            else:
                help_answer = "No PDF context loaded for Gemini."

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Query failed: {str(e)}")
        base_answer = "Error processing request"

    # Cleanup
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Return: base_answer (str), help_answer (str), audio path or None
    return base_answer, help_answer, audio


CUSTOM_CSS = f"""
.gradio-container {{
    background: url("/assets/Background1.png");
    background-size: cover;
    background-position: center;
    min-height: 100vh;
    font-family: 'Segoe UI', sans-serif;
}}

.main-box {{
    background: rgba(255, 255, 255, 0.93) !important;
    backdrop-filter: blur(8px);
    border-radius: 15px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.3);
    padding: 2rem;
    margin: 20px auto;
    max-width: 800px;
}}

.header-section {{
    text-align: center;
    margin-bottom: 2rem;
    padding: 20px;
    background: rgba(255, 255, 255, 0.8);
    border-radius: 12px;
}}

.control-panel {{
    background: rgba(245, 245, 245, 0.7) !important;
    border-radius: 10px;
    padding: 1.5rem;
}}

.response-panel {{
    background: rgba(255, 255, 255, 0.9) !important;
    border-radius: 10px;
    padding: 1.5rem;
    margin-top: 1.5rem;
}}

/* Black background, white text for answer boxes */
.response-card, .help-card {{
    background: #000000 !important; /* Black background */
    color: #ffffff !important; /* White text */
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}}

.response-card ul, .help-card ul {{
    padding-left: 25px;
    margin: 10px 0;
}}

.response-card li, .help-card li {{
    margin-bottom: 8px;
}}

/* Styling for the audio player */
.gradio-container .gr-audio {{
    background: #000000 !important; /* Black background for audio component */
    color: #ffffff !important; /* White text for labels/controls */
    border-radius: 10px;
    padding: 10px;
}}

/* To ensure the audio player controls themselves are visible on a black background, 
   you might need more specific overrides depending on Gradio's internal structure.
   This is a general attempt. */
.gradio-container .gr-audio .waveform,
.gradio-container .gr-audio .play-pause-btn,
.gradio-container .gr-audio .time-display {{
    color: #ffffff !important;
    fill: #ffffff !important; /* For SVG icons */
}}
"""

# Main application
with gr.Blocks(css=CUSTOM_CSS, title="AI Study Assistant") as app:
    with gr.Column(elem_classes="main-box"):
        gr.Markdown("""
        <div class="header-section">
            <h1 style="color: #2c3e50; margin-bottom: 0.5rem;">📚 Learning Assistant</h1>
            <div style="color: #4a5568; font-size: 1.1rem;">
                Your AI-powered study companion
            </div>
        </div>
        """)
        
        with gr.Column(elem_classes="control-panel"):
            status = gr.Markdown(elem_classes="status")
            question = gr.Textbox(label="Your Question", lines=3, placeholder="Ask about any topic in your documents...")
            
            with gr.Row():
                voice_check = gr.Checkbox(label="Voice Response", value=False)
                help_check = gr.Checkbox(label="Additional Help", value=False)
            
            submit_btn = gr.Button("Ask Question", variant="primary")
        
        with gr.Column(elem_classes="response-panel"):
            base_answer = gr.Markdown("## Document Answer\n*Your answer will appear here*", 
                                    elem_classes="response-card")
            help_answer = gr.Markdown("## Additional Insights\n*Enable 'Additional Help' for expert analysis*", 
                                    visible=False,
                                    elem_classes="help-card")
            audio_out = gr.Audio(label="Voice Response", visible=False)
    
    # Initial processing
    initial_status = process_documents()
    status.value = f"<div class='status'>{initial_status}</div>"
    
    # Toggle visibility
    help_check.change(
        lambda x: gr.update(visible=x),
        inputs=help_check,
        outputs=help_answer
    )
    
    voice_check.change(
        lambda x: gr.update(visible=x),
        inputs=voice_check,
        outputs=audio_out
    )
    
    # Handle submission
    submit_btn.click(
        handle_query,
        inputs=[question, voice_check, help_check],
        outputs=[base_answer, help_answer, audio_out]
    )

if __name__ == "__main__":
    # Create required directories
    os.makedirs("data/audio", exist_ok=True)
    os.makedirs("data/texts", exist_ok=True)
    os.makedirs("data/pdfs", exist_ok=True)
    
    app.launch(server_port=7861)