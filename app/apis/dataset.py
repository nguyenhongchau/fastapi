from threading import Thread
from fastapi import APIRouter
from core.models import Dataset
from nets.training_server import flags
import random
from core.utils import download_dataset_file, version_datasets

downloading_urls = []

#APIRouter creates path operations for dataset
dataset_router = APIRouter(
    prefix="/api/datasets",
    tags=["datasets"],
    responses={404: {"description": "Not found"}},
)

@dataset_router.get("/")
async def get_root():
    return {"description": "API path operations for data"}


@dataset_router.post("/add", response_model=Dataset)
def add_dataset(dataset: Dataset):
    if dataset.img_url in downloading_urls:
        return dataset
    
    dataset_id = random.randint(1,1000)
    dataset.dataset_id = dataset_id
    folder = "train" if dataset.trainable else "validation"
    
    downloading_urls.append(dataset.img_url)
    # TODO: It should be put in a queue for downloading large files
    t1 = Thread(target=download_dataset_file, args=(dataset.img_url, folder))
    t2 = Thread(target=download_dataset_file, args=(dataset.label_url, folder))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    downloading_urls.remove(dataset.img_url)
    version_datasets(dataset_id)
    flags["train_dataset_updated"] = True
    
    return dataset


