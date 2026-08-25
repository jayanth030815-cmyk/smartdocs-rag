from typing import List, Dict, Any
import re

class RecursiveCharacterChunker:
    """
    Splits text recursively by trying larger, natural separators first:
    1. Double newlines (paragraphs): "\n\n"
    2. Single newlines (lines): "\n"
    3. Sentence terminators: ". ", "? ", "! "
    4. Spaces (words): " "
    5. Characters: ""
    
    Preserves chunk overlap to prevent cutting critical context at boundaries.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: List[str] = None
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
            
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """
        Splits a single text string into chunks respecting natural language boundaries.
        """
        return self._split(text, self.separators)

    def _split(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        separator = separators[-1]
        new_separators = []

        # Find the highest-level separator present in the text
        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        # Split using the chosen separator
        splits = text.split(separator) if separator else list(text)
        
        good_splits = []
        for s in splits:
            if separator and s:
                good_splits.append(s + separator)
            elif s:
                good_splits.append(s)

        # Merge splits into chunks under chunk_size with chunk_overlap
        current_chunk = []
        current_length = 0

        for piece in good_splits:
            piece_len = len(piece)
            
            if piece_len > self.chunk_size and new_separators:
                # If a single piece is bigger than chunk_size, recursively split it with finer separators
                if current_chunk:
                    merged = "".join(current_chunk).strip()
                    if merged:
                        final_chunks.append(merged)
                    current_chunk = []
                    current_length = 0
                
                sub_chunks = self._split(piece, new_separators)
                final_chunks.extend(sub_chunks)
            elif current_length + piece_len <= self.chunk_size:
                current_chunk.append(piece)
                current_length += piece_len
            else:
                # Chunk is full, save it
                if current_chunk:
                    merged = "".join(current_chunk).strip()
                    if merged:
                        final_chunks.append(merged)
                
                # Apply overlap: keep the tail of the previous chunk
                overlap_pieces = []
                overlap_len = 0
                for p in reversed(current_chunk):
                    if overlap_len + len(p) <= self.chunk_overlap:
                        overlap_pieces.insert(0, p)
                        overlap_len += len(p)
                    else:
                        break
                        
                current_chunk = overlap_pieces + [piece]
                current_length = sum(len(p) for p in current_chunk)

        if current_chunk:
            merged = "".join(current_chunk).strip()
            if merged:
                final_chunks.append(merged)

        return final_chunks

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunks a list of loaded documents and preserves metadata with unique chunk IDs:
        Output:
        [
            {
                "chunk_id": "doc_0_chunk_1",
                "text": "...",
                "metadata": {"source": "report.pdf", "page": 1, "chunk_index": 1, "char_count": 420}
            }
        ]
        """
        all_chunks = []
        chunk_counter = 0

        for doc in documents:
            raw_text = doc["text"]
            base_meta = doc.get("metadata", {})
            text_splits = self.split_text(raw_text)

            for idx, split_text in enumerate(text_splits):
                chunk_counter += 1
                chunk_meta = base_meta.copy()
                chunk_meta["chunk_index"] = idx + 1
                chunk_meta["char_count"] = len(split_text)

                all_chunks.append({
                    "chunk_id": f"chunk_{chunk_counter}",
                    "text": split_text,
                    "metadata": chunk_meta
                })

        return all_chunks
