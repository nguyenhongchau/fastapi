from typing import Optional
from pydantic import BaseModel, Field


class Dataset(BaseModel):
    dataset_id: Optional[int] = None
    model_id: Optional[int] = None
    img_url: str
    label_url: str
    trainable: int
    description: Optional[str] = Field(
        None, title="The description of the dataset", max_length=300
    )

class Training(BaseModel):
    training_id: Optional[int] = None
    status: Optional[str] = None
    validation: Optional[str] = ""
    epoch: Optional[str] = ""

class DataPoint(BaseModel):
    url: str

class InferenceResult(BaseModel):
    pred: int
    prob: float
    model_id: int
    model_version: int 