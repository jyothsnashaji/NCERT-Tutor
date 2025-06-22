from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from collections import defaultdict

from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage

from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import pdb 

# -------------------- FastAPI Setup --------------------
app = FastAPI()

# -------------------- Pydantic Models --------------------
class Query(BaseModel):
    session_id: str
    question: str

class DocumentInfo(BaseModel):
    page: str = ""
    link: str = ""
    snippet: str

class Response(BaseModel):
    answer: str
    retrieved_documents: List[DocumentInfo]

# -------------------- Vector DB --------------------
def load_vector_db(vector_db_path: str):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(persist_directory=vector_db_path, embedding_function=embeddings)

vector_db_path = "vector_db_parent"
vector_db = load_vector_db(vector_db_path)
print("✅ Vector DB loaded")

# -------------------- LLM & Retriever --------------------
groq_api_key = ""
llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-8b-8192")

# Define system prompt
system_prompt = SystemMessage(content="""
    Identify whether the question is related to physics or not.
    If it is physics-related, use tools to ALWAYS retrieve physics context.
    Else respond as a friendly school tutor, ready to help with any academic question.              
""")

# Setup memory and inject system prompt
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


session_memories = {}

def get_memory(session_id: str) -> ConversationBufferMemory:
    if session_id not in session_memories:
        # New session, create fresh memory
        session_memories[session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        # Optionally add your system prompt here for every new memory
        session_memories[session_id].chat_memory.add_message(system_prompt)
    return session_memories[session_id]

# Create contextual retriever with compression
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_db.as_retriever(search_kwargs={"k": 5})
)

# -------------------- Custom Tool With Tracking --------------------
last_retrieved_docs = []

def retrieve_physics_context(query: str) -> str:
    global last_retrieved_docs
    custom_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
        You are a knowledgeable tutor. Use the following **syllabus content** to answer the question.
        Respond clearly, and include examples or problems from the context if possible.
        You are a helpful school tutor. Your job is to explain academic concepts to students in a clear and friendly way.

        - Do NOT mention any tools, sources, or documents.
        - If the context includes examples, use them to make your explanation better.
        - Keep the tone natural and educational.

        Context:
        {context}

        Question: {question}
        Answer:
        """
        )
    qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vector_db.as_retriever(),  
                return_source_documents=True,
                chain_type="stuff",  # most common RAG strategy
                chain_type_kwargs={"prompt": custom_prompt},
                )
    response = qa_chain.invoke(query)
    last_retrieved_docs = response.get("source_documents", [])
    return response

physics_tool = Tool(
    name="retrieve_physics_context",
    func=retrieve_physics_context,
    description="ALWAYS load documents and answer from them if physics related query",
)

# -------------------- Agent Setup --------------------

def get_agent(session_id: str):
    memory = get_memory(session_id)
    agent = initialize_agent(
        tools=[physics_tool],
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        system_prompt=system_prompt
    )
    return agent

# -------------------- API Endpoint --------------------
@app.post("/ask", response_model=Response)
async def ask_question(query: Query):
    try:
        agent = get_agent(query.session_id)
 
        # Run the agent to get the response
        result = agent.run(query.question)

        # Build the document list from last retrieval
        retrieved_docs_info = []
        for doc in last_retrieved_docs:
            metadata = doc.metadata or {}
            retrieved_docs_info.append(DocumentInfo(
                page=str(metadata.get("page", "")),
                link=str(metadata.get("source", "")),
                snippet=doc.page_content[:300]  # First 300 characters
            ))

        return Response(answer=result, retrieved_documents=retrieved_docs_info)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------- Local Run --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
