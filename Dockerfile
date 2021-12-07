FROM pytorch/pytorch

RUN apt-get -y update \
 && apt-get -y install git

COPY . /usr/src/
RUN pip install --no-cache-dir -r /usr/src/requirements.txt
WORKDIR /usr/src/app

EXPOSE 8000
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--reload" ]
