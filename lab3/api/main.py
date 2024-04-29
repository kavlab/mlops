from api_model import InputData
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse, PlainTextResponse
from model import predict_price, r2

MODEL_DIR = "model"

app = FastAPI()


@app.get('/health')
async def health_handler() -> PlainTextResponse:
    """Endpoint для определения состояния сервиса в Docker
    """
    return PlainTextResponse(content='ok', status_code=status.HTTP_200_OK)


@app.get('/test')
async def r2_handler() -> JSONResponse:
    """Endpoint для оценки модели
    """
    return JSONResponse(
        content={'r2': r2()},
        status_code=status.HTTP_200_OK
    )


@app.post('/predict')
async def predict_handler(data: InputData) -> JSONResponse:
    """Endpoint для предсказания цены
    """
    return JSONResponse(
        content={'price': predict_price(data)},
        status_code=status.HTTP_200_OK
    )
