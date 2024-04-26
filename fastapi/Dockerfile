FROM python:3.8 
WORKDIR /code/
COPY . /code/

RUN apt-get update

RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["sh", "run_server.sh"]

