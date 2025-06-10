# bot/pdf_qa_module.py
import os
import io
from dotenv import load_dotenv
import google.generativeai as genai
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
import re


class PDFQASystem:
    """
    A system for answering questions based on content from PDF files
    using the Google Gemini API for explanations.
    """
    def __init__(self):
        """
        Initializes the PDFQASystem, loads the Gemini API key from .env,
        and configures the Gemini API.
        """
        load_dotenv() # Load environment variables from .env file
        self._api_key = os.getenv("GEMINI_API_KEY")

        if not self._api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in your .env file. "
                "Please create a .env file in the same directory as this script and add:\n"
                "GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE\n"
                "You can get your API key from Google AI Studio: https://aistudio.google.com/app/apikey"
            )
        genai.configure(api_key=self._api_key)
        # UPDATED: Changed model name from 'gemini-pro' to 'gemini-1.5-flash'
        self._model = genai.GenerativeModel('gemini-1.5-flash')
        print("PDFQASystem initialized and Gemini API configured.")

    def _extract_text_from_pdf(self, pdf_path):
        """
        (Internal helper) Extracts text from a single PDF file using pdfminer.six.

        Args:
            pdf_path (str): The path to the PDF file.

        Returns:
            str: The extracted text, or an empty string if an error occurs.
        """
        text = ""
        try:
            output_string = io.StringIO()
            with open(pdf_path, 'rb') as fp:
                extract_text_to_fp(fp, output_string, laparams=LAParams(), output_type='text', encoding='utf-8')
            text = output_string.getvalue()
            print(f"  - Successfully extracted text from: {os.path.basename(pdf_path)}")
        except Exception as e:
            print(f"  - Error extracting text from {pdf_path}: {e}")
        return text

    def load_pdfs_from_directory(self, directory_path):
        """
        Loads and concatenates text from all PDF files in a given directory.

        Args:
            directory_path (str): The path to the directory containing PDF files.

        Returns:
            str: A single string containing all extracted text from the PDFs.
                 Returns an empty string if the directory is not found or no PDFs are present.
        """
        all_pdf_text = ""
        if not os.path.isdir(directory_path):
            print(f"Error: Directory not found at '{directory_path}'")
            return ""

        pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]

        if not pdf_files:
            print(f"No PDF files found in '{directory_path}'.")
            return ""

        print(f"Found {len(pdf_files)} PDF files in '{directory_path}'. Extracting text...")
        for pdf_file in pdf_files:
            pdf_path = os.path.join(directory_path, pdf_file)
            all_pdf_text += self._extract_text_from_pdf(pdf_path) + "\n\n" # Add newlines for separation
        print("All PDF text extraction complete.")
        return all_pdf_text

    def get_explained_answer(self, question, context_text):
        """
        Sends a question and context to the Gemini model to get an explained answer.

        Args:
            question (str): The user's question.
            context_text (str): The text extracted from the PDF files.

        Returns:
            str: The generated answer with explanations, or an error message.
        """
        if not context_text:
            return "Error: No context provided from PDF documents. Please load PDFs first."

        try:
            # Construct the prompt to guide Gemini to use the context and provide explanations
            prompt = (
                f"Based on the following context, please answer the question thoroughly "
                f"and provide detailed explanations. If the answer is not directly available "
                f"in the context, state that you cannot find the information.\n\n"
                f"--- Context ---\n{context_text}\n\n"
                f"--- Question ---\n{question}\n\n"
                f"--- Answer with Explanation ---"
            )
            print("Sending request to Gemini API...")
            response = self._model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error communicating with Gemini API: {e}"

# --- Main Application Logic (integrated for direct execution) ---
if __name__ == "__main__":
    try:
        # Initialize the PDF Q&A System
        qa_system = PDFQASystem()

        # Define the directory where your PDF files are located
        pdf_directory = "data/pdfs"

        # Create the directory if it doesn't exist and instruct the user
        if not os.path.exists(pdf_directory):
            os.makedirs(pdf_directory)
            print(f"Created directory: '{pdf_directory}'.")
            print("Please place your PDF files into this directory and run the script again.")
            exit() # Exit if no PDFs are present initially

        # Load text from PDFs using the system's method
        print(f"\nAttempting to load PDFs from '{pdf_directory}'...")
        document_context = qa_system.load_pdfs_from_directory(pdf_directory)

        if not document_context:
            print("No text loaded from PDFs. Please ensure valid PDF files are in the 'data/pdfs' directory.")
            exit() # Exit if no text could be loaded

        print("\nPDF content loaded. You can now ask questions based on the documents.")
        print("Type 'exit' or 'quit' to end the session.")

        # Start the interactive Q&A loop
        while True:
            user_question = input("\nYour question: ").strip()
            if user_question.lower() in ['exit', 'quit']:
                print("Exiting Q&A session. Goodbye!")
                break

            if not user_question:
                print("Please enter a question.")
                continue

            # Get the explained answer using the system's method
            answer = qa_system.get_explained_answer(user_question, document_context)
            print("\n--- Gemini's Answer ---")
            print(answer)
            print("-----------------------")

    except ValueError as ve:
        print(f"Configuration Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
