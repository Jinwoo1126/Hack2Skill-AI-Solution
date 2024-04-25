docker build -f streamlit/Dockerfile -t gen-app streamlit/. &&\
docker build -f fastapi/Dockerfile -t inference-api fastapi/.
