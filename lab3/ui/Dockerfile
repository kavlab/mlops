FROM python:3.11.9-slim

COPY *.py /app/
COPY *.txt /app/

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
