import os
import pdfplumber
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import PyPDFLoader
from langchain.storage import InMemoryStore
import pdb

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                yield page_num, text

def process_pdf(pdf_path, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    documents = []
    for page_num, page_text in extract_text_from_pdf(pdf_path):
        chunks = text_splitter.split_text(page_text)
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": pdf_path,
                    "page": page_num
                }
            )
            documents.append(doc)
    
    print(f"Number of chunks created from {pdf_path}: {len(documents)}")
    return documents

def process_multiple_pdfs(pdf_paths):
    all_documents = []
    pdf_paths = list(Path(pdf_paths).rglob("*.pdf"))  # Recursively find all PDFs

    for pdf_path in pdf_paths:
        documents = process_pdf(str(pdf_path))
        all_documents.extend(documents)
    return all_documents

def load_all_pdfs_from_directory(root_dir: str):
    all_docs = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(dirpath, filename)
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename
                all_docs.extend(docs)
    return all_docs


def get_document_batches(documents, batch_size=1000):
    batches = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batches.append(batch)
        print(f"Batch {len(batches)} created with {len(batch)} documents")
    return batches

def safe_add_documents(batch, retriever, max_batch_size=1000):
    try:
        retriever.add_documents(batch)
    except Exception as e:
        if "greater than max batch size" in str(e) and len(batch) > 1:
            mid = len(batch) // 2
            safe_add_documents(batch[:mid], retriever)
            safe_add_documents(batch[mid:], retriever)
        else:
            raise e


def process_and_store(pdf_paths, retriever_path):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    all_documents = load_all_pdfs_from_directory(pdf_paths)

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

    batches = get_document_batches(all_documents)
    docstore = LocalFileStore("parent_docstore/")
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory="vector_db_parent"
    )

    # === Step 4: Setup Docstore (for parent retrieval) ===
    docstore = InMemoryStore()  # or RedisStore, or SQLStore, etc.

    # === Step 5: Create ParentDocumentRetriever ===
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        parent_splitter=parent_splitter,
        child_splitter=child_splitter
    )
    for batch in batches:
        safe_add_documents(batch, retriever)

    print(f"All chunks from {len(pdf_paths)} PDFs processed and stored in the vector database")

def main():
    pdf_paths = "RAG-Enhanced-NCERT-Tutor/Textbooks"
    vector_db_path = "vector_db_parent"
    
    #process_and_store(pdf_paths, vector_db_path)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
    retriver = vector_db.as_retriever(search_kwargs={"k": 5})
    pdb.set_trace()
    docs = retriver.get_relevant_documents("What is the law of motion?")
    print(docs)

if __name__ == "__main__":
    main()