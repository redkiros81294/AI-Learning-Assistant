import glob
import pdfplumber

"""Load and extract text from all PDFs in a given directory."""
def load_pdfs_from_dir(directory: str):
  
    texts =[]
    for path in glob.glob(f"{directory}/*.pdf"):
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                texts.append({
                    "source": path,
                    "page_number": page.page_number,
                    "text": page.extract_text()
                })
    return texts