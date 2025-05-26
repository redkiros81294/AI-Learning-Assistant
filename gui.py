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
        # Phase 1: Prioritize exact header matches
        header_hits = emb.query(f"HEADER:{question}", top_k=3)
        all_hits = header_hits + emb.query(question, top_k=5)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_hits = []
        for hit in all_hits:
            if hit not in seen:
                seen.add(hit)
                unique_hits.append(hit)

        # Collect and prioritize contexts
        contexts = []
        for doc_id, chunk_id in unique_hits:
            context = emb.get_chunk_text(doc_id, chunk_id)
            if not context:
                continue
            # Boost score for header matches
            if any(q_word.lower() in context.lower() for q_word in ["types", "game", "adversarial"]):
                contexts.insert(0, context)  # Prioritize at start
            else:
                contexts.append(context)

        # Generate answer from combined context
        if contexts:
            combined_context = "\n\n---\n".join(contexts[:100])  # Limit to top 100
            base_answer = _GENERATOR.extract_answer(combined_context, question)
            # Fallback to direct text match
            if "no relevant" in base_answer.lower():
                base_answer = f"Most relevant document chunk:\n{contexts[0][:500]}..."

        # Voice generation with verification (only document answer)
        if use_voice and base_answer and "no relevant" not in base_answer.lower():
            # logger.info(f"Generating audio for: {base_answer[:100]}...")
            logger.info(f"Generating audio for full answer...")
            audio_path = f"data/audio/{hashlib.md5(base_answer.encode()).hexdigest()}.mp3"
            thread = _VOICE.generate_async(base_answer, path=audio_path)
            if thread is not None:
                thread.join(timeout=45)
            if os.path.exists(audio_path):
                audio = audio_path
            else:
                logger.warning("Audio generation timed out")

        # External help (GPT) - always if checked
        if get_help:
    # Use Gemini on PDF context for additional help
            if PDF_CONTEXT:
                help_answer = _PDF_QA.get_explained_answer(question, PDF_CONTEXT)
            else:
                help_answer = "No PDF context loaded for Gemini."

    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        base_answer = "Error processing request"

    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return base_answer, help_answer, audio

def answer_question(question, use_gpt, use_voice):
    embedder = get_embedder()
    generator = AnswerGenerator()
    voice = VoiceGenerator()

    # 1. Retrieve best chunk from documents
    hits = embedder.query(question, top_k=1)
    if not hits:
        return "No matching documents found.", "", None

    doc_id, chunk_id = hits[0]
    context = embedder.get_chunk_text(doc_id, chunk_id)
    if not context:
        return "No matching documents found.", "", None

    # 2. Extract answer from the chunk
    base_answer = generator.extract_answer(context, question)

    # 3. Optionally get supportive info from GPT API
    support = ""
    if use_gpt:
        support = generator.get_external_answer(question)

    # 4. Combine answers for audio
    full_answer = base_answer
    if support:
        full_answer += "\n\nSupportive info:\n" + support

    # 5. Optionally generate audio
    audio_path = None
    if use_voice:
        audio_path = f"data/audio/answer_{hash(question)}.mp3"
        voice.generate_async(full_answer, audio_path)
        # Wait for file to be created (pyttsx3 is synchronous, but if async, add a wait loop)

    return base_answer, support, audio_path

CUSTOM_CSS = f"""
.gradio-container {{
    background: url(/assets/Background.png");
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
"""

# Modified blocks structure
with gr.Blocks(css=CUSTOM_CSS, title="AI Study Assistant") as app:
    with gr.Column(elem_classes="main-box"):
        gr.Markdown("""
        <div class="header-section">
            <h1 style="color: #2c3e50; margin-bottom: 0.5rem;">📚 Learning Assistant</h1>
            <div style="color: #4a5568; font-size: 1.1rem;">
                ◯◯✔✘+✗8✘ | ◯⼜✘+✗◇○○
            </div>
        </div>
        """)
        
        with gr.Column(elem_classes="control-panel"):
            status = gr.Markdown(elem_classes="status")
            question = gr.Textbox(label="Your Question", lines=3)
            
            with gr.Row():
                voice_check = gr.Checkbox(label="Voice Response", value=False)
                help_check = gr.Checkbox(label="Additional Help", value=False)
            
            submit_btn = gr.Button("Ask Question", variant="primary")
        
        with gr.Column(elem_classes="response-panel"):
            base_answer = gr.Textbox(label="Document Answer", lines=4)
            help_answer = gr.Textbox(label="Additional Information", lines=3, visible=False)
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
    app.launch(server_port=7861)