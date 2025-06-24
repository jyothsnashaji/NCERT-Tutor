from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


from fastapi.responses import JSONResponse
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnableSequence 


import re, pprint, os

from dotenv import load_dotenv
load_dotenv()

# print all environment variables for debugging
print("Environment Variables:")
for key, value in os.environ.items():
    print(f"{key}: {value}")

# Ensure the necessary environment variables are set
if not all(key in os.environ for key in ["OPENAI_API_KEY"]):
    raise ValueError("Missing required environment variables: OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

def get_rag_response(query: str, grade: str, subject: str, history=None):
    try:
        print(f"[QUERY] Grade: {grade}, Subject: {subject}, Question: {query}")

        # Setup retriever with metadata filters
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 4,
                "filter": {
                    "grade": grade,
                    "subject": subject.lower(),
                }
            }
        )

        # Use updated LLM import (langchain-openai)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Format conversation history for prompt
        history_text = ""
        if history:
            for turn in history:
                if turn["role"] == "user":
                    history_text += f"User: {turn['content']}\n"
                elif turn["role"] == "assistant":
                    history_text += f"Tutor: {turn['content']}\n"


        # Define prompt with placeholders
        prompt = PromptTemplate.from_template("""
        You are a knowledgeable tutor for grade {grade} in {subject}.
        Below is the conversation so far:
        {history}
        Use only the provided context to answer the question below.
        If the context does not contain the answer,
            and if it is casual talk, respond like a teacher and guide the student towards subject.
            or if it is a question on subject but not within the context, respond it is beyond the scope of the syllabus.

        Context:
        {context}

        Question:
        {input}
        """)

        # Fill static vars (grade, subject) at compile time
        final_prompt = prompt.partial(grade=grade, subject=subject, history=history_text)

        # Build RAG chain
        combine_docs_chain = create_stuff_documents_chain(llm, final_prompt)
        rag_chain: RunnableSequence = create_retrieval_chain(retriever, combine_docs_chain)

        # Run the RAG chain with just `query` as input
        result = rag_chain.invoke({"input": query})
        #print("=== RAG Chain Result ===")
        #pprint.pprint(result)

        # Extract the final answer (always under key "answer")
        raw_answer = result.get("answer", "No answer found.")
        answer = raw_answer.get("answer") if isinstance(raw_answer, dict) else raw_answer

        # Extract sources
        simplified_sources = []
        for doc in result.get("context", []):
            metadata = doc.metadata
            page_content = doc.page_content if hasattr(doc, 'page_content') else "No content available"
            print(f"Source doc metadata: {metadata}")
            simplified_sources.append({
                "filename": metadata.get("source", "Untitled"),
                "page number": metadata.get("page", "#"),
                "page content": page_content[:300]  # Limit to first 200 chars for brevity),
            })

        return {
            "answer": answer,
            "sources": simplified_sources
        }

    except Exception as e:
        print(f"[ERROR] RAG pipeline failed: {e}")
        return {
            "answer": "Sorry, an error occurred while generating your answer.",
            "sources": []
        }

    

def generate_quiz(grade, subject, topic, num_questions=5):

    # Setup retriever with metadata filters
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 1,
            "filter": {
                "grade": grade,
                "subject": subject.lower(),
                "chapter": topic,
            }
        }
    )
    try:
        template = PromptTemplate.from_template("""
        Context:
        {context}
                                                
        Instructions:
        You are a knowledgeable tutor for grade {grade} in {subject}.
        Generate {num_questions} multiple choice questions (MCQs) for grade {grade} and subject {subject} from chapter {topic}
        using only the provided context to answer the question below.
                                                
        If the context does not contain the answer,
            and if it is casual topic, do not generate any quiz and respond like a teacher and guide the student towards subject.
            or if it is a technical topic but not within the context, do not generate any quiz and 
            respond it is beyond the scope of the syllabus.

        If context is enough to generate quiz, then generate the quiz in the following format:
        Each question should have 4 options (A to D) and indicate the correct answer as 'Answer: <option>' after each question.
        Do not include any additional text or explanations, just the questions and options.
        Use the following format for each question:      
        Q1. What is ...?
        A. ...
        B. ...
        C. ...
        D. ...
        Answer: A
        """       )
        prompt = template.partial(
            num_questions=num_questions,
            grade=grade,
            subject=subject,
            topic=topic
        )

        # Build RAG chain
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain: RunnableSequence = create_retrieval_chain(retriever, combine_docs_chain)

        # Run the RAG chain with just `query` as input
        query = f"Generate {num_questions} MCQs for grade {grade} and subject {subject} for topic {topic}."

        result = rag_chain.invoke({"input": query})
        print("=== RAG Chain Result ===")
        pprint.pprint(result)

        return result['answer'] if isinstance(result, dict) and "answer" in result else result
    except Exception as e:
        print(f"[ERROR] Quiz generation failed: {e}")
        return "Error generating quiz."
    
def parse_quiz_text(raw_text: str) -> list[dict]:
    blocks = raw_text.split("Q")[1:]  # Each block starts with "1:", "2:", ...
    parsed_quiz = []
    print(f"[INFO] Found {len(blocks)} quiz blocks to parse.")
    for block in blocks:
        try:
            question_part = block.split("A.")[0].strip()
            options_part = re.findall(r'[A-D]\..+?(?=(?:[A-D]\.|Answer:))', block, re.DOTALL)
            options = {}
            for opt in options_part:
                key = opt.strip()[0]
                val = opt.strip()[2:].strip()
                options[key] = val

            answer_match = re.search(r'Answer:\s*([A-D])', block)
            answer = answer_match.group(1) if answer_match else ""

            parsed_quiz.append({
                "question": question_part,
                "options": options,
                "answer": answer
            })
        except Exception as e:
            print(f"Error parsing block: {block[:30]}... — {e}")

    return parsed_quiz
