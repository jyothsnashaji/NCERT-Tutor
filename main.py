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

# Import the centralized session management functions
from session_manager import get_session_history, get_chat_history

# -------------------- FastAPI Setup --------------------
app = FastAPI()

# -------------------- Models --------------------
class Query(BaseModel):
    session_id: str
    question: str

# -------------------- LLM & Retriever --------------------
llm = ChatGroq(model="llama3-8b-8192")

# -------------------- Destination Chains --------------------
from destination_chains import run_explaination_chain, run_translation_chain, run_exercise_chain, run_general_chain, DestinationChainArgs

def get_destination_chain(arguments: DestinationChainArgs, session_id: str):
    """
    Controller that routes to the correct chain and handles complex flows
    like explain-then-translate.
    """
    destination = arguments.destination
    session_data = get_session_history(session_id)

    if destination == "translate":
        print("--- Destination: TRANSLATE. Orchestrating explain-then-translate flow. ---")
        
        english_answer_to_translate = ""
        last_response = session_data.get("last_response")

        # Check if the router identified a new, specific topic for this query.
        # This is the highest priority.
        if arguments.keyword and arguments.keyword.lower() != 'none':
            print(f"--- New keyword '{arguments.keyword}' found. This takes precedence. ---")
            print("--- Running explanation chain for the new topic first. ---")
            
            # Run the explanation chain for the new keyword.
            explanation_response = run_explaination_chain(arguments, session_id)
            english_answer_to_translate = explanation_response.answer
            
            # IMPORTANT: We save this new explanation as the context for the *next* turn.
            session_data["last_response"] = explanation_response

        # If no new keyword was found, check if it's a follow-up to a previous answer.
        elif last_response and last_response.answer:
            print("--- No new keyword. Assuming follow-up. Translating the previous response. ---")
            english_answer_to_translate = last_response.answer

        # If there's no new keyword AND no previous response, we can't do anything.
        else:
            return Response(
                answer="I'm not sure what you want me to translate. Please ask a question like 'What is photosynthesis?' first.",
                retrieved_documents=[]
            )
        # After running the translation chain, clear the last_response for the next turn
        session_data["last_response"] = None

        # Finally, call the translation chain with the determined English text.
        return run_translation_chain(
            text_to_translate=english_answer_to_translate,
            target_language=arguments.target_language,
            session_id=session_id
        )

    elif destination == "explain":
        # For a simple explanation, we still need to save its result for potential follow-ups.
        explanation_response = run_explaination_chain(arguments, session_id)
        session_data["last_response"] = explanation_response
        return explanation_response

    elif destination == "exercise":
        # An exercise response should also be saved.
        exercise_response = run_exercise_chain(arguments, session_id)
        session_data["last_response"] = exercise_response
        return exercise_response

    elif destination == "general":
        # A general response should also be saved.
        general_response = run_general_chain(arguments, session_id)
        session_data["last_response"] = general_response
        return general_response
        
    else:
        # Fallback for unknown destination
        return run_general_chain(arguments, session_id)
    # -------------------- Destination Chains End--------------------
# -------------------- Router --------------------

router_chains = {}
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

    if session_id not in router_chains:
        router_chains[session_id] = router_prompt | llm 
    chain_with_memory = RunnableWithMessageHistory(
                        router_chains[session_id],
                        get_chat_history,  # function returning a MessageHistory object
                        input_messages_key="input",  # required
                        history_messages_key="chat_history",  # used in your prompt
                    )

    return chain_with_memory

    
# -------------------- Endpoint --------------------

@app.post("/ask", response_model=Response)
async def ask_question(query: Query):
    try:
        # The logic here remains the same, but we add more print statements for clarity
        print("\n" + "="*50)
        print(f"SESSION ID: {query.session_id}, QUESTION: \"{query.question}\"")
        print("="*50)

        # 1. Get the router's decision
        router_chain = get_router_chain(query.session_id)
        router_output = router_chain.invoke({"input": query.question},
                                            config={"configurable": {"session_id": query.session_id}})
        
        print(f"--- Router LLM Output ---\n{router_output.content}\n-------------------------")
        
        parsed_args = output_parser(router_output)
        print(f"--- Parsed Router Args ---\nDestination: {parsed_args.destination}, Keyword: {parsed_args.keyword}\n--------------------------")

        # 2. Execute the destination chain(s)
        result = get_destination_chain(parsed_args, query.session_id)
        print(f"--- Final Result ---\n{result}\n--------------------")

        # 3. Save the final result to session history for the *next* turn
        session_data = get_session_history(query.session_id)
        session_data["last_response"] = result
        print(f"--- Saved final response to session {query.session_id} for next turn. ---")

        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# -------------------- Run --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
