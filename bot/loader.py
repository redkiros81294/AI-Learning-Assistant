# bot/loader.py
import glob
import pdfplumber
from typing import Generator, Dict
import logging

logger = logging.getLogger(__name__)

def load_documents_from_dir(directory: str) -> Generator[Dict, None, None]:
    logger.info(f"Loading documents from {directory}")
    
    pdf_files = glob.glob(f"{directory}/*.pdf")
    txt_files = glob.glob(f"{directory}/*.txt")
    
    logger.info(f"Found {len(pdf_files)} PDFs and {len(txt_files)} TXT files")
    
    for path in pdf_files + txt_files:
        logger.info(f"Processing {path.split('/')[-1]}")
        try:
            if path.endswith(".pdf"):
                with pdfplumber.open(path) as pdf:
                    for page in pdf.pages:
                        page = page.crop(page.bbox) if not page.cropbox else page
                        text = page.extract_text()
                        if text:
                            yield {
                                "source": path,
                                "page_number": page.page_number,
                                "text": text
                            }
            else:
                with open(path, "r") as f:
                    yield {
                        "source": path,
                        "page_number": 1,
                        "text": f.read()
                    }
        except Exception as e:
            logger.error(f"Failed to process {path}: {str(e)}")
