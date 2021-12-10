from fastapi import FastAPI, Depends
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader, APIKey
from starlette.status import HTTP_403_FORBIDDEN
from starlette.responses import RedirectResponse

from transformers import DistilBertTokenizerFast

from model import GeoModel
from model import Coordinate, InputMetrics, PointResponse, GeoRequest
from model import API_DESCRIPTION_STR
from model import MAX_SEQ_LENGTH, BASE_MODEL, TOKEN_MODEL, API_VERSION

geomodel = GeoModel(
    model_string=BASE_MODEL,
    tokenizer_string=TOKEN_MODEL,
    tokenizer_class=DistilBertTokenizerFast,
    max_seq_length=MAX_SEQ_LENGTH)

API_KEYS = [
    "geo",
]
API_KEY_NAME = "access_token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

app = FastAPI(
    title="GEOFYR",
    version=API_VERSION,
    description=API_DESCRIPTION_STR
)


async def validate_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header in API_KEYS:
        return api_key_header
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Could not validate credentials"
        )


@app.get("/", include_in_schema=False)
def docs_redirect():
    return RedirectResponse("/docs")


@app.post("/inference/point", response_model=PointResponse)
def inference_point(request: GeoRequest,
                    api_key: APIKey = Depends(validate_api_key)
                    ):
    '''
    Infers the geographic location of a given text as a single point.
    For security reasons this endpoint only accepts the text as encrypted SSL payload. 
    '''
    # Split text to tokens
    split_text = request.text.split(" ")

    # Slice tokens and calc token metrics
    input_tokens = len(split_text)
    inference_tokens = min(input_tokens, MAX_SEQ_LENGTH)
    inference_text = str(" ").join(split_text[:inference_tokens])
    unused_tokens = len(split_text[inference_tokens:])

    # Inference
    geomodel.forward(inference_text)
    point_coordinate = geomodel.predict_point()

    # Compose respond
    coordinate = Coordinate(lat=point_coordinate[0], lon=point_coordinate[1])
    input_metrics = InputMetrics(
        input_tokens=input_tokens,
        inference_tokens=inference_tokens,
        unused_tokens=unused_tokens
    )
    return PointResponse(coordinate=coordinate, input_metrics=input_metrics)

