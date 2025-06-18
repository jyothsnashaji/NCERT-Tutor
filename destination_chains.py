from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from prompts import explain_prompt, general_prompt, quiz_prompt
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
import pdb,re
from langchain_groq import ChatGroq
from pydantic import BaseModel
from langchain_core.runnables import RunnableSequence 

from dotenv import load_dotenv
load_dotenv()


explain_chains = {}
session_histories = {}
translation_chains = {}
exercise_chains = {}
general_chains = {}

def get_session_history(session_id):
    if session_id not in session_histories:
        session_histories[session_id] = InMemoryChatMessageHistory()
    return session_histories[session_id]

class Quiz(BaseModel):
    question: str
    options: List[str]
    answer: str

class DocumentInfo(BaseModel):
    page: str
    file: str
    snippet: str

class Response(BaseModel):
    answer: Optional[str]
    retrieved_documents: Optional[List[DocumentInfo]]
    quiz: Optional[List[Quiz]] = None

class ExplainDestinationChain():
    def __init__(self, session_id):
        self.session_id = session_id
        self.llm =  ChatOpenAI(model="gpt-4o-mini")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = FAISS.load_local("/Users/jshaji/Library/CloudStorage/OneDrive-Cisco/IISc/Deep Learning/NCERT-Tutor/vector_store_physics", self.embeddings, allow_dangerous_deserialization=True)
        self.retriever = self.vectorstore.as_retriever()
        self._chain = explain_prompt | self.llm
      
                                
    def get_rag_response(self, query: str, grade: str = "", subject: str = ""):
        try:
            print(f"[QUERY] Grade: {grade}, Subject: {subject}, Question: {query}")


            # Step 1: Final prompt with static variables filled in
            final_prompt = explain_prompt.partial(grade=grade, subject=subject)

            # Step 2: Combine retrieved documents into string context
            combine_docs_chain = create_stuff_documents_chain(self.llm, final_prompt)
            # Step 3: Create RAG chain
            rag_chain = create_retrieval_chain(self.retriever, combine_docs_chain)

            # Step 4: Add chat memory
            rag_chain_with_memory = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history"
            )

            # Step 5: Run it
            result = rag_chain_with_memory.invoke(
                {"input": query, "chat_history": get_session_history(self.session_id).messages},
                config={"configurable": {"session_id": self.session_id}}
            )

            print("=== RAG Chain Result ===")
            print(result)

            # Extract the final answer (always under key "answer")
            raw_answer = result.get("answer", "No answer found.")
            answer = raw_answer.get("answer") if isinstance(raw_answer, dict) else raw_answer

            # Extract sources
            simplified_sources = []
            for doc in result.get("context", []):
                metadata = doc.metadata
                page_content = doc.page_content if hasattr(doc, 'page_content') else "No content available"
                print(f"Source doc metadata: {metadata}")
                simplified_sources.append(DocumentInfo(
                    page=str(metadata.get("page", "#")),
                    file=str(metadata.get("source", "Untitled")),
                    snippet=page_content[:300]  # First 300 characters
                ))
                    

            return Response(
                answer=answer,
                retrieved_documents=simplified_sources
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[ERROR] RAG pipeline failed: {e}")
            return Response(
                answer="Sorry, an error occurred while generating your answer.",
                retrieved_documents=[]
            )

class TranslationDestinationChain():
    def __init__(self, session_id):
        self.session_id = session_id
        # Define translation chain here if needed

class ExerciseDestinationChain():
    def __init__(self, session_id):
        self.session_id = session_id
        self.llm =  ChatOpenAI(model="gpt-4o-mini")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = FAISS.load_local("/Users/jshaji/Library/CloudStorage/OneDrive-Cisco/IISc/Deep Learning/NCERT-Tutor/vector_store_physics", self.embeddings, allow_dangerous_deserialization=True)
        self.retriever = self.vectorstore.as_retriever()
        self._chain = quiz_prompt | self.llm

    def get_rag_response(self, query: str, num_questions: int = 5, grade: str = "", subject: str = ""):
        retriever = self.vectorstore.as_retriever(
        # search_kwargs={
        #     "k": 1,
        #     "filter": {
        #         "grade": grade,
        #         "subject": subject.lower(),
        #         "chapter": topic,
        #     }
        # }
    )
        try:
            prompt = quiz_prompt.partial(
                num_questions=num_questions,
                grade=grade,
                subject=subject,
                topic=query
            )

            # Build RAG chain
            combine_docs_chain = create_stuff_documents_chain(self.llm, prompt)
            rag_chain: RunnableSequence = create_retrieval_chain(retriever, combine_docs_chain)

            # Run the RAG chain with just `query` as input
            input = f"Generate {num_questions} MCQs for grade {grade} and subject {subject} for topic {query}."

            result = rag_chain.invoke({"input": input, "chat_history": get_session_history(self.session_id).messages},
                                      config={"configurable": {"session_id": self.session_id}})
            print("=== RAG Chain Result ===")
            print(result)

            return result['answer'] if isinstance(result, dict) and "answer" in result else result
        except Exception as e:
            print(f"[ERROR] Quiz generation failed: {e}")
            return "Error generating quiz."
    
    def parse_quiz_text(self,text: str) -> List[Quiz]:
        # Split the text into individual questions
        questions = re.split(r'\n(?=Q\d+\.)', text.strip())
        quizzes = []

        for q in questions:
            # Match question number, question, options and answer
            match = re.match(
                r"Q\d+\.\s*(.*?)\s*"
                r"A\.\s*(.*?)\s*"
                r"B\.\s*(.*?)\s*"
                r"C\.\s*(.*?)\s*"
                r"D\.\s*(.*?)\s*"
                r"Answer:\s*([A-D])", q, re.DOTALL
            )
            if match:
                question_text = match.group(1).strip()
                options = [match.group(i).strip() for i in range(2, 6)]
                answer_index = "ABCD".index(match.group(6))
                quizzes.append(Quiz(
                    question=question_text,
                    options=options,
                    answer=options[answer_index]
                ))
            else:
                print(f"Failed to parse: {q}")

        return quizzes
            
class GeneralDestinationChain():
    def __init__(self, session_id):
        self.session_id = session_id
        self.llm = ChatGroq(model="llama3-8b-8192") 
        self._chain = general_prompt | self.llm
        
        # Define general chain here if needed

class DestinationChainArgs():
    def __init__(self, destination, keyword, target_language, query):
        self.destination = destination
        self.keyword = keyword
        self.target_language = target_language
        self.query = query



def get_explaination_chain(session_id):
    if session_id not in explain_chains:
        explain_chains[session_id] = ExplainDestinationChain(session_id)
    return explain_chains[session_id]

def get_translation_chain(session_id):
    if session_id not in translation_chains:
        translation_chains[session_id] = TranslationDestinationChain(session_id)
    return translation_chains[session_id]

def get_exercise_chain(session_id):
    if session_id not in exercise_chains:
        exercise_chains[session_id] = ExerciseDestinationChain(session_id)
    return exercise_chains[session_id]

def get_general_chain(session_id):
    if session_id not in general_chains:
        general_chains[session_id] = GeneralDestinationChain(session_id)
    return general_chains[session_id]

def run_general_chain(arguments, session_id):
    chain = get_general_chain(session_id)
    answer =  chain._chain.invoke(
        {"input": arguments.query, "chat_history": get_session_history(session_id).messages},
        config={"configurable": {"session_id": session_id}}
    )
    return Response(answer=answer.content, retrieved_documents=[]) # Assuming no sources for general queries


def run_explaination_chain(arguments: DestinationChainArgs, session_id: str):
    """
    Run the explaination chain with the provided arguments and session ID.
    Inputs: DestinationChainArgs object containing:
        - destination: The type of destination (e.g., "explain", "translate", etc.)
        - keyword: The keyword to search for in the context
        - target_language: The language to translate to (if applicable)
        - question: The question or query to be answered
    Outputs: Response object containing:
        - answer: The generated answer from the explaination chain
        - retrieved_documents: List of DocumentInfo objects containing source information
    """
    chain = get_explaination_chain(session_id)
    return chain.get_rag_response(arguments.keyword)

def run_translation_chain(arguments, session_id):
    """Run the translation chain with the provided arguments and session ID.
    Inputs: DestinationChainArgs object containing:
        - destination: The type of destination (e.g., "explain", "translate", etc.)
        - keyword: The keyword to search for in the context
        - target_language: The language to translate to (if applicable)
        - question: The question or query to be answered
    Outputs: Response object containing:
        - answer: The generated answer from the translation chain
        - retrieved_documents: List of DocumentInfo objects containing source information
    """
    return Response(
        answer="Translation functionality is not yet implemented.",
        retrieved_documents=[]
    )  # Placeholder response until translation chain is implemented

def run_exercise_chain(arguments, session_id):
    """Run the exercise chain with the provided arguments and session ID.
    Inputs: DestinationChainArgs object containing:
        - destination: The type of destination (e.g., "explain", "translate", etc.)
        - keyword: The keyword to search for in the context
        - target_language: The language to translate to (if applicable)
        - question: The question or query to be answered
    Outputs: List of Response object containing:
        - answer: The generated answer from the exercise chain
        - retrieved_documents: List of DocumentInfo objects containing source information
        - quiz: Quiz object containing generated questions and answers
        
    """
    chain = get_exercise_chain(session_id)
    rag_response = chain.get_rag_response(query=arguments.keyword)
    quizzes =  chain.parse_quiz_text(rag_response) 
    return Response(
        answer="",
        retrieved_documents=[],
        quiz=quizzes
    )  # Assuming no sources for quiz generation