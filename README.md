

## Usage

1. Install requirements
   
2. Generate Embedding and create vector store
   ```
   python3 build_vector_store.py
   ```

3. Start the FastAPI backend:
   ```
   uvicorn main:app --reload
   ```

4. Launch the Streamlit UI:
   ```
   streamlit run streamlit_app.py
   ```

5. Open your web browser and navigate to the Streamlit app URL (typically `http://localhost:8501`)

## Textbooks
Textbooks from various grades and subjects are stored in specific hierarchy inside the data folder. They are parsed for embedding and stored in vector stores for RAG to make use of these documents. The data directory contains textbooks in this format,

* data/
  * grade_xx/
    * subject
      * cc_chapter1name
      * cc_chapter2name

xx - grade number, cc - chapter number

## Data Ingestion and Chunking
The script "build_vector_store.py" scans the 'data/' directory for PDF files. For each PDF, it extracts relevant metadata such as grade, subject, and chapter from the filepath. The text content of each PDF is then split into manageable chunks to facilitate efficient retrieval and embedding.

## Embedding Generation
Each text chunk is converted into a high-dimensional vector representation using OpenAI's embedding model (text-embedding-3-small). This process enables semantic search, allowing the system to retrieve content based on meaning rather than exact keyword matches.

## Vector Store Creation
All embedded document chunks are stored in a FAISS vector store, which is optimized for fast similarity search. This vector store serves as the backbone for the Retrieval-Augmented Generation (RAG) pipeline, enabling the system to quickly find and retrieve the most relevant content in response to user queries.

The resulting vector store is saved locally in the 'vector_store' directory for efficient loading and reuse during inference.

## To Do
1. Add Translation Support
2. Update exerise generation with T5
3. Add evaluation dataset and metrics

## Demo

![Explaination Response](demo/Explaination.jpg)
![RAG Response](demo/sources.jpg)
![Exercise Response](demo/quiz.jpg)
