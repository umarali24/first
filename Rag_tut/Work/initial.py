from langchain_community.document_loaders import PyMuPDFLoader

def load_pdf(file_path):
    # Load the PDF document using PyMuPDF
    doc = PyMuPDFLoader(file_path)
    
    # Extract text from each page and concatenate it
    doc = doc.load()
    return doc

if __name__ == "__main__":
    pdf_data = load_pdf("C:\Work\AI\Rag_tut\Data\Chapter-01 environment.pdf")
    print(pdf_data[0].metadata)
    print(pdf_data[0].page_content)
    # for page in pdf_data:
    #     print(page.metadata)
    #     print(page.page_content)
    