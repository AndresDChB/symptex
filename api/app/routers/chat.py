import logging
from typing import AsyncGenerator

from fastapi import (APIRouter, Depends)
from fastapi.responses import StreamingResponse, PlainTextResponse
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.db import get_db
from app.db.models import ChatSession, ChatMessage, PatientFile, AnamDoc
from app.utils.pdf_utils import anamdoc_to_dict
from chains.chat_chain import build_symptex_model
from chains.eval_chain import eval_history
from chains.formatting import format_patient_details

# Set up logging
logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)

router = APIRouter()


# Chat request schema
class ChatRequest(BaseModel):
    message: str
    model: str
    condition: str
    talkativeness: str
    patient_file_id: int
    session_id: str

# Rate request schema
class RateRequest(BaseModel):
    messages: list

# Chat endpoint
@router.post("/chat")
async def chat_with_llm(request: ChatRequest, db: Session = Depends(get_db)):
    """Endpoint to chat with the LLM"""
    logger.debug("Received chat request: %s", request)
   
    # Validate message, condition and talkativeness first
    if not request.message:
        logger.error("Empty message received")
        raise PlainTextResponse("Message cannot be empty", status_code=400)
    if request.model not in ["gemma-3-27b-it", "llama-3.3-70b-instruct", "llama-3.1-sauerkrautlm-70b-instruct", "qwq-32b", "mistral-large-instruct", "qwen3-235b-a22b"]:
        logger.error("Invalid model: %s", request.model)
        raise PlainTextResponse(f"Invalid model: {request.model}", status_code=400)
    if request.condition not in ["default", "alzheimer", "schwerhörig", "verdrängung"]:
        logger.error("Invalid condition: %s", request.condition)
        raise PlainTextResponse(f"Invalid condition: {request.condition}", status_code=400)
    if request.talkativeness not in ["kurz angebunden", "ausgewogen", "ausschweifend"]:
        logger.error("Invalid talkativeness: %s", request.talkativeness)
        raise PlainTextResponse(f"Invalid talkativeness: {request.talkativeness}", status_code=400)
    
    # Get patient profile from database
    patient_file = db.query(PatientFile).filter(PatientFile.id == request.patient_file_id).first()
    if not patient_file:
        return PlainTextResponse("Patient not found", status_code=404)
    patient_details = format_patient_details(patient_file)

    patient_doc_rows = db.query(AnamDoc).filter(AnamDoc.patient_file_id == patient_file.id).all()
    patient_doc_md = [anamdoc_to_dict(row) for row in patient_doc_rows]

    #todo remove this
    print(f"Patient metadata: {patient_doc_md}")

    #check that the LLM is aware of the new context
    # Create or get chat session
    session = db.query(ChatSession).filter(
        ChatSession.id == request.session_id
    ).first()
    if not session:
        session = ChatSession(
            id=request.session_id,
            patient_file_id=request.patient_file_id
        )
        db.add(session)
        db.commit()
        db.refresh(session)

     # Get previous messages from database
    previous_messages = []
    chat_history = db.query(ChatMessage).filter(
        ChatMessage.session_id == session.id
    ).order_by(ChatMessage.timestamp.asc()).all()
    
    for msg in chat_history:
        if msg.role == "user":
            previous_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "patient":
            previous_messages.append(AIMessage(content=msg.content))


    # Store message
    message = ChatMessage(
        session_id=session.id,
        role="user",
        content=request.message
    )
    db.add(message)
    db.commit()

    try:
        llm_response = ""

        # Stream response and store LLM message
        async def generate_and_store():
            nonlocal llm_response
            try:
                messages = previous_messages + [HumanMessage(content=request.message)]
                async for chunk in stream_response(
                    model=request.model,
                    condition=request.condition,
                    talkativeness=request.talkativeness,
                    patient_details=patient_details,
                    patient_doc_md = patient_doc_md,
                    session_id=request.session_id,
                    previous_messages=messages
                ):
                    #todo extend with custom chatevent to distinguish between text chunk or state
                    llm_response += chunk
                    yield chunk
                
                # After streaming is complete, store LLM message
                llm_message = ChatMessage(
                    session_id=session.id,
                    role="patient",
                    content=llm_response
                )
                db.add(llm_message)
                db.commit()
            finally:
                db.close()

        return StreamingResponse(
            generate_and_store(), 
            media_type="text/plain"
        )
    except Exception as e:
        logger.error("Error in chat_with_llm endpoint: %s", str(e))
        return PlainTextResponse("Internal server error", status_code=500)
    
# Reset endpoint
@router.post("/reset/{session_id}")
async def reset_memory(session_id: str, db: Session = Depends(get_db)):
    """Reset the LangChain memory for a specific session"""
    try:
        # Delete messages from db
        db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
        # Delete the session itself
        db.query(ChatSession).filter(ChatSession.id == session_id).delete()
        db.commit()
        return PlainTextResponse(f"Chat data deleted for session {session_id}", status_code=200)
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {str(e)}")
        db.rollback()
        return PlainTextResponse("Error deleting session", status_code=500)
    finally:
        db.close()
    
# Evaluation endpoint
@router.post("/eval")
async def eval_chat(request: RateRequest):
    # Convert frontend messages to LangChain messages
    from langchain_core.messages import HumanMessage, AIMessage

    async def generate_eval():
        try:
            lc_messages = []
            for msg in request.messages:
                if msg["role"] == "user":
                    lc_messages.append(HumanMessage(content=msg["output"]))
                elif msg["role"] == "patient":
                    lc_messages.append(AIMessage(content=msg["output"]))

            # Stream evaluation chunks
            async for chunk in eval_history(lc_messages):
                yield chunk
            
        except Exception as e:
            logger.error(f"Error generating evaluation: {str(e)}")
            yield f"Entschuldigung, es ist ein Fehler aufgetreten: {str(e)}"

    try:
        return StreamingResponse(
            generate_eval(),
            media_type="text/plain"
        )
    except Exception as e:
        logger.error(f"Error rating chat: {str(e)}")
        return PlainTextResponse("Error rating chat", status_code=500)


async def stream_response(
    model: str, 
    condition: str, 
    talkativeness: str, 
    patient_details: str,
    patient_doc_md: list[dict],
    session_id: str,
    previous_messages: list
) -> AsyncGenerator[str, None]:
    """
    Stream responses from the symptex_model.

    Args:
        message (str): The input message from the user.
        model (str): The model to use for generating the response.
        condition (str): The medical condition to simulate.
        talkativeness (str): The level of talkativeness for the response.
        patient_details (str): Details about the patient.
        patient_doc_md (list[dict]): Metadata of patient documents.
        session_id (str): The ID of the chat session.
        previous_messages (list): A list of previous messages in the chat.

    Returns:
        str: The response message from the LLM.
    """

    initial_state = {
        "messages": previous_messages,
        "model": model,
        "condition": condition,
        "talkativeness": talkativeness,
        "patient_details": patient_details,
        "patient_doc_md": patient_doc_md,
    }
    #todo remove this
    logger.info("Initial state: %s", initial_state)
    symptex_model = build_symptex_model(initial_state)
    try:
        async for mode, chunk in symptex_model.astream(
            initial_state,
            stream_mode=["messages", "values"],
        ):
            if mode == "messages":
                #todo figure out why even with /nothink the thing is thinking
                msg, metadata = chunk
                if msg.content and not isinstance(msg, HumanMessage):
                    yield msg.content

            elif mode == "values":
                # full graph state after this step
                final_state = chunk
                print(final_state)
                #todo test if tool is working then handle this
    except Exception as e:
        logger.error("Error while streaming response: %s", str(e))
        yield f"Entschuldigung, es ist ein Fehler aufgetreten: {str(e)}"
