from fastapi import FastAPI, status
from fastapi.responses import JSONResponse, Response
from model import predict_price, r2
from api_model import InputData


MODEL_DIR = "model"

app = FastAPI()


@app.get('/health')
async def health_handler() -> Response:
    """
    """
    return Response(status_code=status.HTTP_200_OK)


@app.get('/test')
async def r2_handler() -> Response:
    """
    """
    return JSONResponse(
        content={'r2': r2()},
        status_code=status.HTTP_200_OK
    )


@app.post('/predict')
async def predict_handler(data: InputData) -> JSONResponse:
    """
    """
    return JSONResponse(
        content={'price': predict_price(data)},
        status_code=status.HTTP_200_OK
    )
