
# 🤖 AI Learning Assistant

A sophisticated AI-powered document question-answering system that transforms your lecture notes and academic materials into an intelligent, interactive learning companion. Built with modern RAG (Retrieval-Augmented Generation) architecture, this system combines local document processing with external AI services to provide accurate, contextual answers with optional voice synthesis.

## 🌟 Key Features

### 📚 **Multi-Format Document Processing**
- **PDF Intelligence**: Advanced PDF text extraction using `pdfplumber` [1](#0-0) 
- **Text File Support**: Direct processing of plain text lecture notes
- **Semantic Chunking**: Intelligent document segmentation using spaCy NLP models

### 🔍 **Advanced Search & Retrieval**
- **Vector Embeddings**: Powered by sentence-transformers for semantic understanding [2](#0-1) 
- **FAISS Indexing**: Lightning-fast similarity search across your document corpus
- **Context-Aware Retrieval**: Finds relevant information even with paraphrased queries

### 🧠 **Hybrid AI Generation**
- **Local Processing**: Transformers-based answer generation for privacy
- **Enhanced Explanations**: Google Gemini integration for comprehensive responses [3](#0-2) 
- **Fallback Architecture**: Graceful degradation when external services are unavailable

### 🎵 **Multi-Engine Voice Synthesis**
- **Primary TTS**: pyttsx3 for reliable cross-platform speech synthesis [4](#0-3) 
- **Advanced TTS**: Coqui TTS engine for high-quality voice output
- **Accessibility**: Full audio responses for enhanced learning accessibility

### 🎨 **Modern Web Interface**
- **Gradio Framework**: Clean, responsive web interface [5](#0-4) 
- **Real-time Processing**: Live document indexing and query processing
- **User-Friendly**: Intuitive design for seamless interaction

## 🏗️ System Architecture

The AI Learning Assistant implements a modular, scalable architecture:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Gradio UI     │───▶│  Document Loader │───▶│  Preprocessor   │
│   (gui.py)      │    │  (loader.py)     │    │ (preprocessor.py)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Voice Generator │◀───│ Answer Generator │◀───│   Embedder      │
│   (voice.py)    │    │ (generator.py)   │    │ (embedder.py)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Core Components

| Component | Purpose | Key Technologies |
|-----------|---------|------------------|
| **Document Loader** | PDF/Text ingestion | `pdfplumber`, `pdfminer.six` |
| **Preprocessor** | Text segmentation | `spaCy`, NLP pipelines |
| **Embedder** | Vector generation | `sentence-transformers`, `FAISS` |
| **Generator** | Answer synthesis | `transformers`, Google Gemini |
| **Voice Engine** | Speech synthesis | `pyttsx3`, Coqui TTS |

## 📁 Project Structure

```
AI-Learning-Assistant/
├── 🎯 gui.py                 # Main application entry point
├── 📋 requirements.txt       # Python dependencies
├── 🔐 .env                   # API keys and configuration
├── 
├── 🤖 bot/                   # Core processing modules
│   ├── loader.py            # Document ingestion (PDF/TXT)
│   ├── preprocessor.py      # Text chunking and NLP
│   ├── embedder.py          # Vector embeddings and search
│   ├── generator.py         # Answer generation pipeline
│   ├── voice.py             # Text-to-speech synthesis
│   ├── nlp.py              # Natural language processing
│   └── trainer.py          # Optional GPT-2 fine-tuning
│
└── 📚 data/                  # Data storage and cache
    ├── pdfs/               # Input PDF documents
    ├── texts/              # Input text files
    ├── embeddings/         # FAISS vector indices
    ├── chunk_cache/        # Processed text segments
    └── audio/              # Generated speech files
```

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8+ 
- 4GB+ RAM (8GB recommended for large document sets)
- Optional: CUDA-compatible GPU for faster processing

### Installation

1. **Clone and Setup**
   ```bash
   git clone https://github.com/redkiros81294/AI-Learning-Assistant.git
   cd AI-Learning-Assistant
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Configure API Access**
   ```bash
   # Create .env file
   echo "GEMINI_API_KEY=your-gemini-api-key-here" > .env
   ```

4. **Add Your Documents**
   ```bash
   # Place your files
   cp your-lecture-notes.pdf data/pdfs/
   cp your-text-notes.txt data/texts/
   ```

5. **Launch Application**
   ```bash
   python gui.py
   ```
   
   Access the interface at: http://127.0.0.1:7861 [6](#0-5) 

## 💡 Usage Examples

### Basic Question Answering
- **Query**: "What is the A* search algorithm?"
- **Response**: Document-based answer with relevant context from your lecture notes

### Enhanced Explanations
- Enable "Additional Help" for Gemini-powered detailed explanations [7](#0-6) 
- Get both document-specific answers and broader conceptual explanations

### Voice Learning
- Check "Voice Response" for audio answers [8](#0-7) 
- Perfect for auditory learners and accessibility needs

## 🔧 Advanced Features

### Custom Model Fine-tuning
Fine-tune GPT-2 on your specific lecture content: [9](#0-8) 
```bash
python bot/trainer.py
```

### Performance Optimization
- **GPU Acceleration**: Automatic CUDA detection for faster processing
- **Caching System**: Intelligent caching of embeddings and processed chunks
- **Batch Processing**: Efficient handling of large document collections

## 🛠️ Technology Stack

### Core AI/ML Stack
- **PyTorch 2.3.0**: Deep learning framework
- **Transformers 4.41.2**: Pre-trained language models  
- **Sentence-Transformers 2.7.0**: Semantic embeddings
- **FAISS 1.7.4**: Vector similarity search
- **spaCy 3.8.7**: Advanced NLP processing

### Integration & Interface
- **Gradio 4.36.1**: Modern web interface
- **Google Generative AI 0.3.0**: Gemini API integration
- **pyttsx3 2.98**: Cross-platform TTS
- **Coqui TTS 0.22.0**: Advanced voice synthesis

## 🔍 Sample Educational Content

The repository includes demonstration materials covering advanced AI topics: [10](#0-9) 

- **Search Algorithms**: A* search, minimax, alpha-beta pruning
- **Constraint Satisfaction**: CSP formulation, backtracking, arc consistency  
- **Decision Theory**: Expectimax search, utility theory, uncertainty handling

## 🐛 Troubleshooting

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| **Gemini API Errors** | Verify API key in `.env` file [11](#0-10)  |
| **TTS Not Working** | Install system dependencies: `sudo apt-get install espeak` [12](#0-11)  |
| **spaCy Model Missing** | Run: `python -m spacy download en_core_web_sm` [13](#0-12)  |
| **CUDA Errors** | System automatically falls back to CPU processing [14](#0-13)  |

## 🤝 Contributing

We welcome contributions! Areas for enhancement:
- Additional document format support (DOCX, EPUB)
- Multi-language processing capabilities  
- Advanced visualization features
- Performance optimizations

## 📄 License & Credits

Built with powerful open-source technologies:
- [Google Gemini](https://ai.google.dev/) - Advanced AI capabilities [15](#0-14) 
- [HuggingFace Transformers](https://huggingface.co/transformers/) - ML models [16](#0-15) 
- [Gradio](https://gradio.app/) - Web interface framework [17](#0-16) 
- [spaCy](https://spacy.io/) - Industrial-strength NLP [18](#0-17) 

---

**Transform your learning experience with AI-powered document intelligence.** 🎓✨
```

