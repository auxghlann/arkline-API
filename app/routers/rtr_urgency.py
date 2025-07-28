from fastapi import APIRouter, HTTPException
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR
from pydantic import BaseModel
from app.client.arkline_ai_urgency import ArklineAI
from typing import Optional

client = ArklineAI()
router = APIRouter(
    prefix="/urgency",
   tags=["Urgency"]
)

class UrgencyRequest(BaseModel):
    subject: Optional[str]
    message: str

class UrgencyResponse(BaseModel):
    urgency: str

@router.post("/get", tags=["Urgency"])
def get_urgency(request: UrgencyRequest) -> UrgencyResponse:
    try:
        response = client.get_response(request.subject, request.message)
        
        # The client now has fallback logic, so response should always have urgency
        if not response or 'urgency' not in response:
            # This should rarely happen now, but provide a final fallback
            return UrgencyResponse(urgency="Others")
            
        return UrgencyResponse(urgency=response['urgency'])
    except HTTPException:
        raise
    except Exception as e:
        # Log the error for debugging but still return a valid response
        print(f"Urgency classification error: {str(e)}")
        return UrgencyResponse(urgency="Others")