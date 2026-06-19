# chunking.py

import fitz
import re
from unstructured.partition.text import partition_text
from unstructured.chunking.title import chunk_by_title
from langchain_core.documents import Document


class Chunker:

    def __init__(self, file_path: str, max_characters: int = 500,
                 new_after_n_chars: int = 400, overlap: int = 50):
        self.file_path  = file_path
        self.filename   = file_path.split("\\")[-1]
        self.max_characters    = max_characters
        self.new_after_n_chars = new_after_n_chars
        self.overlap           = overlap


    def extract_pages(self) -> list[dict]:
        """
        Extract text page by page using PyMuPDF.
        Returns list of {page_number, text} dicts.
        """
        doc = fitz.open(self.file_path)
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append({
                    "page_number": i + 1,
                    "text"       : text
                })
        doc.close()
        return pages


    def _is_noise(self, text: str) -> bool:
        """
        Detect noise elements:
        - Page headers/footers
        - Figure/Table labels
        - Very short fragments
        - Lone page numbers
        """
        text = text.strip()

        noise_patterns = [
            r"^Chapter\s+\d+\s*\|",        # "Chapter 1 | Environmental Setting"
            r"^Fig\.?\s+\d+\.\d+",         # "Fig 1.2 Pakistan:..."
            r"^Figure\s+\d+",              # "Figure 1 ..."
            r"^Table\s+\d+",               # "Table 1.1 ..."
            r"^\d+\s*$",                   # lone page numbers
            r"^Environmental Setting\s*$", # standalone repeated header
        ]

        for pattern in noise_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True

        if len(text) < 40:
            return True

        return False


    def _is_references_start(self, text: str) -> bool:
        """Detect the start of the References/Bibliography section."""
        return bool(re.match(r"^references\s*$", text.strip(), re.IGNORECASE))


    def get_elements(self) -> list:
        """
        Extract and partition text elements page by page.
        Filters noise and stops at References section.

        Also manually tracks the current heading ("Title" category elements
        detected by partition_text) and attaches it to every element as a
        custom metadata attribute `section_heading`. This is needed because
        the installed version of `unstructured` has no built-in `section`
        field on ElementMetadata -- chunk_by_title does NOT populate a
        section/breadcrumb automatically, so we have to track it ourselves.
        """
        pages = self.extract_pages()
        all_elements = []
        in_references = False
        current_section = None

        for page in pages:
            elements = partition_text(
                text=page["text"],
                include_page_breaks=False,
            )

            for element in elements:
                text = element.text.strip()

                # Stop at references section
                if self._is_references_start(text):
                    in_references = True

                if in_references:
                    continue

                # Update current heading BEFORE noise filtering -- headings
                # are usually short (e.g. "1.1 Physical Setting"), so the
                # length-based noise check below would otherwise discard
                # them before we ever get a chance to read them.
                if element.category == "Title":
                    current_section = text

                # Skip noise, but never drop a real Title element just
                # because it's short -- short length is normal for headings.
                if element.category != "Title" and self._is_noise(text):
                    continue

                # Manually attach page number since partition_text doesn't track it
                element.metadata.page_number = page["page_number"]

                # Manually attach the current heading (custom attribute -- not a
                # built-in ElementMetadata field, but plain attribute assignment
                # works and survives into chunk.metadata.orig_elements later)
                element.metadata.section_heading = current_section

                all_elements.append(element)

        return all_elements


    def chunk_documents(self) -> list[Document]:
        """
        Full pipeline:
        1. Extract text page by page
        2. Partition into structural elements
        3. Filter noise + references
        4. Chunk by title (structure-aware)
        5. Return LangChain Documents with metadata
        """

        # Step 1 + 2 + 3: Get clean structural elements
        elements = self.get_elements()

        # Step 4: Structure-aware chunking
        # include_orig_elements=True ensures each chunk keeps a reference to
        # the original pre-chunk elements it was built from -- this is how we
        # recover the section_heading we attached in get_elements().
        chunks = chunk_by_title(
            elements,
            max_characters       = self.max_characters,
            new_after_n_chars    = self.new_after_n_chars,
            overlap              = self.overlap,
            include_orig_elements = True,
        )

        # Step 5: Convert to LangChain Documents
        langchain_chunks = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "source"      : self.file_path,
                "filename"    : self.filename,
                "category"    : chunk.category,
                "page_number" : chunk.metadata.page_number,
                "chunk_index" : i,
            }

            # Recover the section/heading breadcrumb from the original
            # elements this chunk was built from.
            section = self._get_section_for_chunk(chunk)
            if section:
                metadata["section"] = section

            content = chunk.text.strip()
            if not content:
                continue

            langchain_chunks.append(Document(
                page_content=content,
                metadata=metadata
            ))

        return langchain_chunks


    def _get_section_for_chunk(self, chunk) -> str | None:
        """
        Recover the heading/section breadcrumb for a chunk produced by
        chunk_by_title. We attached `section_heading` to every original
        element in get_elements(); chunk.metadata.orig_elements holds those
        original elements, so we just read it off the first one.
        """
        orig_elements = getattr(chunk.metadata, "orig_elements", None)
        if not orig_elements:
            return None

        for el in orig_elements:
            section = getattr(el.metadata, "section_heading", None)
            if section:
                return section

        return None


    def print_chunks(self, chunks: list[Document]):
        for i, chunk in enumerate(chunks):
            print(f"\n{'=' * 50}")
            print(f"Chunk    : {i + 1}")
            print(f"Category : {chunk.metadata.get('category')}")
            print(f"Page     : {chunk.metadata.get('page_number')}")
            print(f"Section  : {chunk.metadata.get('section', 'N/A')}")
            print(f"Metadata : {chunk.metadata}")
            print(f"Content  :\n{chunk.page_content}")
            print(f"Length   : {len(chunk.page_content)} chars")


if __name__ == "__main__":
    file_path = r"C:\Work\AI\Rag_tut\Data\Chapter-01 environment.pdf"

    chunker = Chunker(file_path)
    chunks  = chunker.chunk_documents()

    print(f"Total chunks: {len(chunks)}")
    chunker.print_chunks(chunks)