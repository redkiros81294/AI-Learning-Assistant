# bot/generator.py
from transformers import pipeline
import logging
import torch
from dotenv import load_dotenv
import os
import re


# Add Gemini import
import google.generativeai as genai

load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

logger = logging.getLogger(__name__)

class AnswerFormatter:
    def structure_response(self, text: str, question: str) -> str:
        """Organize answers based on content patterns"""
        # Identify list patterns
        if self._is_list(text):
            return self._format_as_list(text)
        # Identify step-by-step processes
        elif "step" in question.lower() or "procedure" in question.lower():
            return self._format_as_steps(text)
        # Identify definitions
        elif "what is" in question.lower() or "define" in question.lower():
            return self._format_as_definition(text)
        # Default to paragraph format
        else:
            return self._format_as_paragraph(text)
    
    def _is_list(self, text: str) -> bool:
        """Check if text contains list indicators"""
        list_indicators = ["\n- ", "\n* ", "\n1. ", "\n• ", ":", ";"]
        return any(indicator in text for indicator in list_indicators)
    
    def _format_as_list(self, text: str) -> str:
        """Convert text to markdown list"""
        lines = text.split('\n')
        formatted = []
        for line in lines:
            if line.startswith(('-', '*', '•', '→')) or (line.strip() and line.strip()[0].isdigit()):
                formatted.append(f"- {line.strip()}")
            elif ':' in line or ';' in line:
                parts = re.split(r'[:;]', line, 1)
                if len(parts) > 1:
                    formatted.append(f"- **{parts[0].strip()}**: {parts[1].strip()}")
            else:
                formatted.append(line)
        return "\n".join(formatted)
    
    def _format_as_steps(self, text: str) -> str:
        """Format step-by-step instructions"""
        steps = re.split(r'\n\d+\.|\n-|\n•|\n→', text)
        return "\n".join([f"{i+1}. {step.strip()}" for i, step in enumerate(steps) if step.strip()])
    
    def _format_as_definition(self, text: str) -> str:
        """Highlight key definitions"""
        parts = text.split('.', 1)
        if len(parts) > 1:
            return f"**Definition**: {parts[0]}\n\n**Explanation**: {parts[1]}"
        return f"**Definition**: {text}"
    
    def _format_as_paragraph(self, text: str) -> str:
        """Break into readable paragraphs"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        grouped = []
        for i in range(0, len(sentences), 3):
            group = '. '.join(sentences[i:i+3])
            if not group.endswith('.'):
                group += '.'
            grouped.append(group)
        return "\n\n".join(grouped)

class AnswerGenerator:
    def __init__(self):
        self.qa_pipe = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=0 if torch.cuda.is_available() else -1
        )
        self.formatter = AnswerFormatter()
        
        # Configure Gemini
        if GEMINI_KEY:
            genai.configure(api_key=GEMINI_KEY)
    
    def extract_answer(self, context: str, question: str) -> str:
        try:
            result = self.qa_pipe(question=question, context=context)
            if result['score'] > 0.15 and result['answer'].strip():
                return self.formatter.structure_response(result['answer'], question)
            else:
                return self.formatter.structure_response(context[:500] + "...", question)
        except Exception as e:
            return self.formatter.structure_response(context[:500] + "...", question)

    def get_external_answer(self, question: str) -> str:
        try:
            model = genai.GenerativeModel("models/gemini-pro")
            response = model.generate_content(question)
            # For the latest google-generativeai, use response.text or response.candidates[0].content.parts[0].text
            if hasattr(response, "text"):
                return self.formatter.structure_response(response.text, question)
            elif hasattr(response, "candidates"):
                return self.formatter.structure_response(response.candidates[0].content.parts[0].text, question)
            else:
                return str(response)
        except Exception as e:
            logger.error(f"Gemini failed: {str(e)}")
            return "Could not fetch additional information"