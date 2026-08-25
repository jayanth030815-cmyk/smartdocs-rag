import os
from typing import List, Dict, Any
from pathlib import Path
import pypdf

class DocumentLoader:
    """
    Loads and extracts text with metadata from various file formats:
    - PDF (.pdf) with per-page tracking
    - Plain Text (.txt)
    - Markdown (.md)
    - Microsoft Word (.docx)
    """

    @staticmethod
    def load_file(file_path: str) -> List[Dict[str, Any]]:
        """
        Loads a file and returns a list of page/section dictionaries:
        [
            {"text": "...", "metadata": {"source": "filename.pdf", "page": 1, ...}}
        ]
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        filename = path.name

        if ext == ".pdf":
            return DocumentLoader._load_pdf(file_path, filename)
        elif ext in [".txt", ".md", ".py", ".json", ".csv"]:
            return DocumentLoader._load_text(file_path, filename)
        elif ext == ".docx":
            return DocumentLoader._load_docx(file_path, filename)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported: .pdf, .txt, .md, .docx")

    @staticmethod
    def _load_pdf(file_path: str, filename: str) -> List[Dict[str, Any]]:
        pages = []
        try:
            reader = pypdf.PdfReader(file_path)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                clean_text = text.strip()
                if clean_text:
                    pages.append({
                        "text": clean_text,
                        "metadata": {
                            "source": filename,
                            "page": page_num,
                            "total_pages": len(reader.pages)
                        }
                    })
        except Exception as e:
            raise RuntimeError(f"Error reading PDF {filename}: {str(e)}")
        return pages

    @staticmethod
    def _load_text(file_path: str, filename: str) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
        
        if not content:
            return []
            
        return [{
            "text": content,
            "metadata": {
                "source": filename,
                "page": 1,
                "total_pages": 1
            }
        }]

    @staticmethod
    def _load_docx(file_path: str, filename: str) -> List[Dict[str, Any]]:
        try:
            import docx
            doc = docx.Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text.strip())
            text = "\n\n".join(full_text)
            return [{
                "text": text,
                "metadata": {
                    "source": filename,
                    "page": 1,
                    "total_pages": 1
                }
            }]
        except ImportError:
            raise ImportError("python-docx is required to read .docx files. Install with 'pip install python-docx'.")
