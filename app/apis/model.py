import random
from fastapi import APIRouter
from core.models import Training, InferenceResult, DataPoint
from threading import Thread
from nets.training_server import do_train, flags, get_training_metrics
from nets.serving_server import do_inference

#APIRouter creates path operations for model
model_router = APIRouter(
    prefix="/api/models",
    tags=["models"],
    responses={404: {"description": "Not found"}},
)

@model_router.get("/")
async def get_root():
    return {"description": "API path operations for models"}


@model_router.post("/{model_id}/train", response_model=Training)
def retrain_model(model_id: int):
    if flags["is_training"]:
        return get_training_metrics("training")
    elif not flags["train_dataset_updated"]:
        return get_training_metrics("latest")
    else:
        flags["is_training"] = True
        flags["train_dataset_updated"] = False
        t = Thread(target=do_train)
        t.start()
        flags["training_id"] = random.randint(1,1000)
        training = Training(training_id=flags["training_id"], status="started")
        return training


@model_router.post("/{model_id}/inference", response_model=InferenceResult)
def inference_single_data(model_id: int, data: DataPoint, model_version: int = -1):
    pred, prob = do_inference(data.url)
    if pred is not None:
        res = InferenceResult(pred=pred, prob=prob, model_id=model_id, model_version=model_version)
    else:
        res = InferenceResult(pred=-1, prob=-1, model_id=model_id, model_version=model_version)
    return res


@model_router.get("/training/last", response_model=Training)
async def read_training_status():
    if flags["is_training"]:
        return get_training_metrics("training")
    else:
        return get_training_metrics("stopped")