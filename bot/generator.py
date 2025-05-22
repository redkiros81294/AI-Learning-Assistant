from transformers import pipeline
import logging
import torch
from dotenv import load_dotenv
import os

# Add Gemini import
import google.generativeai as genai

load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

logger = logging.getLogger(__name__)

class AnswerGenerator:
    def __init__(self):
        self.qa_pipe = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=0 if torch.cuda.is_available() else -1
        )
        # Configure Gemini
        if GEMINI_KEY:
            genai.configure(api_key=GEMINI_KEY)
    
    def extract_answer(self, context: str, question: str) -> str:
        try:
            result = self.qa_pipe(question=question, context=context)
            if result['score'] > 0.15 and result['answer'].strip():
                return result['answer']
            else:
                return context[:500] + "..."
        except Exception as e:
            return context[:500] + "..."

    def get_external_answer(self, question: str) -> str:
        try:
            model = genai.GenerativeModel("models/gemini-pro")
            response = model.generate_content(question)
            # For the latest google-generativeai, use response.text or response.candidates[0].content.parts[0].text
            if hasattr(response, "text"):
                return response.text
            elif hasattr(response, "candidates"):
                return response.candidates[0].content.parts[0].text
            else:
                return str(response)
        except Exception as e:
            logger.error(f"Gemini failed: {str(e)}")
            return "Could not fetch additional information"