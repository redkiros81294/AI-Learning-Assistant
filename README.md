# AI Learning Assistant

An AI-powered assistant that answers questions from your lecture notes (PDF or TXT), provides concise document-based answers, can generate additional explanations using Google Gemini, and optionally reads answers aloud using TTS.  
Accessible via a modern Gradio web interface.

---

## Features

- **Document QA:** Answers questions using your own lecture notes (PDF/TXT).
- **Semantic Search:** Embeds and indexes content for fast, relevant retrieval.
- **Additional Help:** Optionally augments answers using Google Gemini (Gemini-Pro).
- **Voice Output:** Converts document answers to speech (pyttsx3 TTS).
- **Fine-tuning:** Optional GPT-2 fine-tuning on your notes.
- **Modern UI:** Clean Gradio web interface.

---

## Project Structure

```
.
├── .env
├── .gitignore
├── README.md
├── requirements.txt
├── gui.py
├── bot/
│   ├── __init__.py
│   ├── embedder.py
│   ├── generator.py
│   ├── loader.py
│   ├── nlp.py
│   ├── preprocessor.py
│   ├── trainer.py
│   └── voice.py
└── data/
    ├── pdfs/
    ├── texts/
    ├── audio/
    ├── chunk_cache/
    └── embeddings/
```

---

## Setup Instructions

### 1. Clone the Repository

```sh
git clone https://github.com/redkiros81294/AI-Learning-Assistant.git
cd AI-Learning-Assistant
```

### 2. Create and Activate a Virtual Environment

```sh
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```sh
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root with the following (use your own Gemini API key):

```
GEMINI_API_KEY=your-gemini-api-key-here
```

> **Note:** OpenAI is no longer required for additional help; Gemini is now used.

### 5. Add Your Lecture Files

- Place your PDF files in `data/pdfs/`
- Place any plain text files in `data/texts/`

### 6. Run the Application

```sh
python gui.py
```

- The Gradio web interface will open in your browser (default: http://127.0.0.1:7861).

---

## Usage

1. **Process Documents:** On first run, documents are automatically processed and indexed.
2. **Ask Questions:** Enter your question in the UI.
3. **Voice Response:** Check the box to get an audio answer (document answer only).
4. **Additional Help:** Check the box to get a Gemini-powered explanation in addition to the document answer.

---

## Fine-tuning GPT-2 (Optional)

You can fine-tune GPT-2 on your notes for custom generation:

```sh
python bot/trainer.py
```
Edit `bot/trainer.py` for your data and parameters.

---

## Troubleshooting

- **Gemini API errors:** Ensure your API key is correct and you have access to Gemini-Pro.
- **TTS issues:** Make sure your system supports `pyttsx3` (Linux: may require `espeak` or `libespeak`).
- **spaCy errors:** Ensure `en_core_web_sm` is downloaded.
- **CUDA errors:** If you don't have a GPU, the app will fall back to CPU automatically.

---

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies.

---

## Credits

- [Google Gemini](https://ai.google.dev/)
- [spaCy](https://spacy.io/)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [Gradio](https://gradio.app/)
- [pyttsx3](https://pyttsx3.readthedocs.io/)

---



