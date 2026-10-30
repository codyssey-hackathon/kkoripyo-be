from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.models import Analysis

router = APIRouter(prefix="/analyses", tags=["analyses"])


class AnalysisCreate(BaseModel):
    handle: str = Field(pattern=r"^[A-Za-z0-9._]{1,30}$", examples=["example_user"])


class AnalysisOut(BaseModel):
    id: int
    subject_handle: str
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


@router.post("", response_model=AnalysisOut, status_code=201)
def create_analysis(body: AnalysisCreate, db: Session = Depends(get_db)) -> Analysis:
    analysis = Analysis(subject_handle=body.handle.lower())
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis


@router.get("/{analysis_id}", response_model=AnalysisOut)
def get_analysis(analysis_id: int, db: Session = Depends(get_db)) -> Analysis:
    analysis = db.get(Analysis, analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="분석을 찾을 수 없습니다.")
    return analysis
