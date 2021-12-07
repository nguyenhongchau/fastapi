import uvicorn
from fastapi import FastAPI
from apis.dataset import dataset_router
from apis.model import model_router

app = FastAPI()

@app.get('/')
def index():
    return {
        "message": "Hello world"
    }

app.include_router(dataset_router)
app.include_router(model_router)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    print("running")