from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_groq import ChatGroq
from destination_chains import Response
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

import pdb 
from prompts import router_prompt
import json
from dotenv import load_dotenv
load_dotenv()


# -------------------- FastAPI Setup --------------------
app = FastAPI()

# -------------------- Models --------------------
class Query(BaseModel):
    session_id: str
    question: str

# -------------------- LLM & Retriever --------------------
llm = ChatGroq(model="llama3-8b-8192")

# -------------------- Memory --------------------

# Store sessions
session_histories = {}

def get_session_history(session_id):
    if session_id not in session_histories:
        session_histories[session_id] = InMemoryChatMessageHistory()
    return session_histories[session_id]
# -------------------- Destination Chains --------------------
from destination_chains import run_explaination_chain, run_translation_chain, run_exercise_chain, run_general_chain, DestinationChainArgs
def get_destination_chain(arguments, session_id: str):
    if arguments.destination == "explain":
        return run_explaination_chain(arguments, session_id)
    elif arguments.destination == "translate":
        return run_translation_chain(arguments, session_id)
    elif arguments.destination == "exercise":
        return run_exercise_chain(arguments, session_id)
    elif arguments.destination == "general":
        return run_general_chain(arguments, session_id)


# -------------------- Router --------------------

router_chain = {}
memory_store = {}


def output_parser(router_result):
    content = router_result.content.replace("None", "null")
    parsed_result = json.loads(content)
    next_inputs = DestinationChainArgs(
        destination=parsed_result["destination"],
        keyword=parsed_result["next_inputs"]["keyword"],
        target_language=parsed_result["next_inputs"]["target_language"],
        query=parsed_result["next_inputs"]["question"]
    )
    return next_inputs

def get_router_chain(session_id: str):

    if session_id not in router_chain:
        router_chain[session_id] = router_prompt | llm 
    chain_with_memory = RunnableWithMessageHistory(
                        router_chain[session_id],
                        get_session_history,  # function returning a MessageHistory object
                        input_messages_key="input",  # required
                        history_messages_key="chat_history",  # used in your prompt
                    )

    return chain_with_memory

    
# -------------------- Endpoint --------------------
@app.post("/ask", response_model=Response)
async def ask_question(query: Query):
    try:
        router_chain = get_router_chain(query.session_id)
        answer = router_chain.invoke({"input": query.question},
                                    config={"configurable": {"session_id": query.session_id}})
        print(answer)
        answer = output_parser(answer)
        print(f"Parsed Answer: {answer}")
        result = get_destination_chain(answer, query.session_id)
        print(result)

        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# -------------------- Run --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
