from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from collections import defaultdict

from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.agents import tool, AgentExecutor, ConversationalAgent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pdb 

# -------------------- FastAPI Setup --------------------
app = FastAPI()

# -------------------- Models --------------------
class Query(BaseModel):
    session_id: str
    question: str

class DocumentInfo(BaseModel):
    page: str
    link: str
    snippet: str

class Response(BaseModel):
    answer: str
    retrieved_documents: List[DocumentInfo]

# -------------------- Vector DB & Retriever --------------------
def load_vector_db(vector_db_path: str):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(persist_directory=vector_db_path, embedding_function=embeddings)

vector_db_path = "/Users/jshaji/Library/CloudStorage/OneDrive-Cisco/IISc/Deep Learning/vector_db4"
vector_db = load_vector_db(vector_db_path)
print("✅ Vector DB loaded")

# -------------------- LLM & Retriever --------------------
groq_api_key = ""
llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-8b-8192")


tools = []

# -------------------- Memory --------------------
memory_store = defaultdict(lambda: ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
))

def get_shared_memory(session_id: str):
    return memory_store[session_id]

# -------------------- Prompts --------------------
technical_prompt = PromptTemplate(
    template="""
You are a friendly AI tutor for school students who answers queries over NCERT textbooks.

Explain in simple language. Use only the context to answer the user's question.

{chat_history}
{input}
Assistant:""",
    input_variables=["input", "chat_history"]
)

casual_prompt = PromptTemplate(
    template="You are a friendly Tutor for school children. Identify if the student needs help with their studies. {input}",
    input_variables=["input"]
)

# -------------------- Agent Setup --------------------
technical_agents = {}
casual_chains = {}

def get_technical_agent(session_id: str):
    if session_id not in technical_agents:
        memory = get_shared_memory(session_id)
        agent = ConversationalAgent.from_llm_and_tools(
            llm=llm,
            tools=tools,
            system_message=technical_prompt,
            input_variables=["input", "chat_history", "context"]
        )
        executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True
        )
        technical_agents[session_id] = executor
    return technical_agents[session_id]

def get_casual_chain(session_id: str):
    if session_id not in casual_chains:
        memory = get_shared_memory(session_id)
        chain = LLMChain(llm=llm, prompt=casual_prompt, memory=memory)
        casual_chains[session_id] = chain
    return casual_chains[session_id]

# -------------------- Router --------------------
router_prompt = PromptTemplate(
    template="""
You are a smart router that classifies student questions as either:
- "technical" → if the question is about school subjects like physics, math, chemistry, definitions, experiments, etc.
- "casual" → if it's a greeting, personal chat, or general question.

Handle follow up to technical queries also as technical.

Respond with one word: technical or casual.

Question: {input}
Answer:""",
    input_variables=["input"]
)

router_chain = LLMChain(llm=llm, prompt=router_prompt)

def classify_query_type(question: str) -> str:
    result = router_chain.run(question).strip().lower()
    return result if result in ["technical", "casual"] else "casual"

# -------------------- Endpoint --------------------
@app.post("/ask", response_model=Response)
async def ask_question(query: Query):
    try:
        query_type = classify_query_type(query.question)

        if query_type == "technical":

            retriever = vector_db.as_retriever(search_kwargs={"k": 5})
            docs = retriever.get_relevant_documents(query.question)
            context = "\n\n".join(
                f"[Page {doc.metadata.get('page', 'N/A')}] {doc.page_content}"
                for doc in docs
            )
            agent = get_technical_agent(query.session_id)
            input_with_context = f"Context:\n{context}\n\nQuestion: {query.question}"
            answer = agent.run(input_with_context)
        else:
            chain = get_casual_chain(query.session_id)
            answer = chain.run(query.question)
            docs = []

        doc_info = [
            DocumentInfo(
                page=str(doc.metadata.get("page", "N/A")),
                link=doc.metadata.get("source", "N/A"),
                snippet=doc.page_content
            )
            for doc in docs
        ]

        return Response(answer=answer, retrieved_documents=doc_info)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------- Run --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
