from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette import status

from app.client.arkline_ai_chat import ArklineAIChat

router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)

# Initialize the chat instance
chat_instance = ArklineAIChat()

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

@router.post("/response", response_model=ChatResponse)
async def rtr_chat(request: ChatRequest):
    try:
        # Initialize vectorstore if not already done
        if not chat_instance.vectorstore:
            chat_instance.process_document()
        
        answer = chat_instance.answer_question(request.question)  
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))   