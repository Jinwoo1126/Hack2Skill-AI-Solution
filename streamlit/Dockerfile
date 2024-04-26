FROM google/cloud-sdk:latest
WORKDIR /code/
COPY . /code/

RUN apt-get update
RUN apt-get install -y python3.8 & \
    ln -s /usr/bin/python3.8 /usr/bin/python

RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["streamlit", "run", "home.py"]
