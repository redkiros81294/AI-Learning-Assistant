# AI Learning Assistant

This project is an AI-powered learning assistant that can answer questions from your PDF lecture notes, generate text answers, and optionally provide voice responses.

## Features

- Extracts and preprocesses text from PDF files
- Embeds and indexes lecture content for retrieval
- Fine-tunes GPT-2 on your notes (optional)
- Answers questions using retrieval or generative models
- Converts answers to speech using Dia TTS
- Simple Gradio web interface

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
    └── pdfs/
        ├── Lecture 2.pdf
        ├── Lecture 3 Informed Search.pdf
        ├── Lecture 4 Adversarial Search.pdf
        ├── Lecture 5 Uncertainity and Utility.pdf
        └── Lecture 6 CSP.pdf
```

## Setup Instructions

### 1. Clone the Repository

```sh
git clone https://github.com/redkiros81294/AI-Learning-Assistant.git
cd  Learning Assistant
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
```

### 4. Download spaCy Model

```sh
python -m spacy download en_core_web_sm
```

### 5. Set Up Environment Variables

- Edit `.env` and add your OpenAI API key:
  ```
  OPENAI_API_KEY="sk-..."
  ```

### 6. Add Your PDF Files

- Place your lecture PDFs in the `data/pdfs/` directory.

### 7. Run the Application

```sh
python gui.py
```

- This will launch a Gradio web interface in your browser.

## Notes

- The first run will preprocess your PDFs and create `data/combined_notes.txt`.
- Fine-tuning GPT-2 is optional and can be triggered via [`bot/trainer.py`](bot/trainer.py).
- For TTS, the Dia model will be downloaded automatically.

## Troubleshooting

- If you encounter issues with dependencies, ensure your Python version is 3.8 or higher.
- For GPU acceleration, ensure PyTorch is installed with CUDA support.

## License

MIT License
