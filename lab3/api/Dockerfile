FROM python:3.11.9-slim

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY *.py /app/
COPY *.txt /app/

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt

RUN python3 ./data_preprocessing.py
RUN python3 ./model_preparation.py

EXPOSE 8080

HEALTHCHECK CMD curl --fail http://localhost:8080/health

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
