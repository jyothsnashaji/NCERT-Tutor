import os
import fitz  # PyMuPDF
from typing import List
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from pathlib import Path
from PIL import Image
from io import BytesIO

def extract_images_with_captions_from_pdf(pdf_path: str):
    doc = fitz.open(pdf_path)
    results = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)

        print(f"Page {page_num} — Found {len(images)} image(s)")
        for img_index, img in enumerate(images):
            print(f"  Image {img_index}: xref={img[0]}, width={img[2]}, height={img[3]}, bpc={img[4]}")
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # Heuristic: get all text from page (could refine this for fig. captions)
            caption = page.get_text().strip()[:300]  # Keep it short

            results.append({
                "page": page_num,
                "image_bytes": image_bytes,
                "image_ext": image_ext,
                "caption": caption,
                "source_file": pdf_path
            })
        

    return results


def extract_from_directory(directory: str):
    all_image_data = []
    pdf_paths = list(Path(directory).rglob("*.pdf"))
    for pdf_path in pdf_paths:
        image_data = extract_images_with_captions_from_pdf(pdf_path)
        all_image_data.extend(image_data)
    return all_image_data


def create_documents(image_data: List[dict]) -> List[Document]:
    documents = []
    for data in image_data:
        doc = Document(
            page_content=data["caption"],
            metadata={
                "image_ext": data["image_ext"],
                "image_bytes": data["image_bytes"],
                "page": data["page"],
                "source_file": data["source_file"]
            }
        )
        documents.append(doc)
    return documents


def build_vector_store(directory: str, save_path: str = None):
    image_data = extract_from_directory(directory)
    docs = create_documents(image_data)

    # Use HuggingFace embeddings (e.g., all-MiniLM-L6-v2)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(docs, embeddings)

    if save_path:
        vectorstore.save_local(save_path)
        print(f"Vectorstore saved at {save_path}")

    return vectorstore

def query_vector_store(query: str, vectorstore_path: str, k: int = 3):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    results = vectorstore.similarity_search(query, k=k)

    for i, doc in enumerate(results):
        print(f"Result {i+1}:")
        print("Caption:", doc.page_content)
        print("Source File:", doc.metadata.get("source_file"))
        print("Page:", doc.metadata.get("page"))

        image_bytes = doc.metadata.get("image_bytes")
        image_ext = doc.metadata.get("image_ext", "png")
        image = Image.open(BytesIO(image_bytes))
        image.show(title=f"Result {i+1}")

        print("---")

if __name__ == "__main__":
    # Provide the path to your PDF directory
    pdf_directory = "NCERT-Tutor/Textbooks"
    output_path = "./vectorstore_images/image_index"
    #build_vector_store(pdf_directory, output_path)
    test_query = "Show me an image related to Newton's laws"
    query_vector_store(test_query, output_path)
