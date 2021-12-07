## Create Docker
```console
docker build -t nguyen:fastapi .
```
## Run service
### Run in Docker container
```console
docker run -p 8000:8000 -t -i nguyen:fastapi
```
### Run locally 
```console
pip install -r requirements.txt
cd app
uvicorn main:app --host 0.0.0.0 --reload
```
## Request samples
Run inference first to check the accuracy of model is low, almost random prediction. Then can add new dataset, train and test reference again.
### Inference Datapoint
```console
curl -X POST http://127.0.0.1:8000/api/models/1/inference -H 'Content-Type: application/json' -d '{"url":"https://raw.githubusercontent.com/nguyenhongchau/fastapi/main/data/datapoint/0.jpg"}'
```
You can change the image name from 0.jpg to 9.jpg. 
### Train with new dataset
```console
curl -X POST http://127.0.0.1:8000/api/models/1/train
```
### GET training status
```console
curl -X GET http://127.0.0.1:8000/api/models/training/last
```

### POST new dataset
Sync request to know when new dataset updated, so please wait :). New dataset is about 8Mb.
```console
curl -X POST http://127.0.0.1:8000/api/datasets/add -H 'Content-Type: application/json' -d '{"img_url":"https://raw.githubusercontent.com/nguyenhongchau/fastapi/main/data/dataset/a_images","label_url":"https://raw.githubusercontent.com/nguyenhongchau/fastapi/main/data/dataset/a_labels","trainable": 1}'
```
You can change a_ => b_ for second sample dataset, trainable 1 => 0 to add new dataset in validation.