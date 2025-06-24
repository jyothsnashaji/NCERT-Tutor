"""
This script builds a vector store from PDF documents in the 'data/' directory.
It processes each PDF, extracts metadata (grade, subject, chapter), splits the text into chunks,
and saves the resulting vector store to 'vector_store/'.
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

import os

def load_and_split_all(data_dir="data/"):
    """
    This function loads and splits all PDF documents in the specified directory.
    It extracts metadata such as grade, subject, and chapter from the file paths,
    and returns a list of Document objects with the text split into manageable chunks.
    """
    all_docs = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)

                # Extract grade, subject, chapter from path
                parts = file_path.split(os.sep)
                try:
                    grade = parts[-3].replace("grade_", "")  
                    subject = parts[-2].lower()              
                    chapter = os.path.splitext(file)[0].lower()  
                    print(f"Grade: {grade}, Subject: {subject}, Chapter: {chapter}")
                except IndexError:
                    print(f"[WARNING] Skipping improperly structured path: {file_path}")
                    continue

                print(f"[INFO] Processing → Grade {grade}, Subject {subject}, Chapter {chapter}")

                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()

                    # Add metadata to each page
                    for doc in docs:
                        doc.metadata.update({
                            "grade": grade,
                            "subject": subject,
                            "chapter": chapter
                        })

                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    split_docs = splitter.split_documents(docs)
                    all_docs.extend(split_docs)

                except Exception as e:
                    print(f"[ERROR] Failed to process {file_path}: {e}")
    
    return all_docs

if __name__ == "__main__":
    print("[INFO] Loading and splitting all documents...")
    all_docs = load_and_split_all("data/")
    print(f"Total chunks created: {len(all_docs)}")
    
    if not all_docs:
        print("[WARNING] No documents found. Please check your data directory.")
    else:
        print(f"[INFO] Creating vector store with {len(all_docs)} chunks...")
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
        vectorstore = FAISS.from_documents(all_docs, embeddings)
        vectorstore.save_local("vector_store")
        print("[SUCCESS] Vector store saved to 'vector_store/'")
