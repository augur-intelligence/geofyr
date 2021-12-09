from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import Security, FastAPI, HTTPException
from fastapi.security.api_key import APIKeyQuery, APIKeyHeader, APIKey
from starlette.status import HTTP_403_FORBIDDEN

from transformers import DistilBertTokenizerFast

from model import GeoModel
from model import Coordinate, Meta, InputMetrics, PointResponse, GeoRequest
from model import API_DESCRIPTION_STR
from model import MAX_SEQ_LENGTH, BASE_MODEL, TOKEN_MODEL, API_VERSION, MODEL_VERSION

geomodel = GeoModel(
    model_string=BASE_MODEL,
    tokenizer_string=TOKEN_MODEL,
    tokenizer_class=DistilBertTokenizerFast,
    max_seq_length=MAX_SEQ_LENGTH)

API_KEYS = [
    "geo",
]
API_KEY_NAME = "access_token"

api_key_query = APIKeyQuery(name=API_KEY_NAME, auto_error=False)
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

app = FastAPI(
    title = "GEOFYR",
    version=API_VERSION,
    description=API_DESCRIPTION_STR
)


async def get_api_key(
    api_key_query: str = Security(api_key_query),
    api_key_header: str = Security(api_key_header),
):

    if api_key_query in API_KEYS:
        return api_key_query
    elif api_key_header in API_KEYS:
        return api_key_header
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials"
        )
        

@app.get("/")
async def root():
    return {"msg": "Welcome to GEOFYR. Visit /docs for documentation <3"}
    

@app.post("/inference/point", response_model = PointResponse)
async def inference_point(request: GeoRequest, api_key: APIKey = Depends(get_api_key)):
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
        input_tokens = input_tokens,
        inference_tokens = inference_tokens,
        unused_tokens = unused_tokens
    )
    return PointResponse(coordinate=coordinate, input_metrics=input_metrics)

