from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import Security, FastAPI, HTTPException
from fastapi.security.api_key import APIKeyQuery, APIKeyHeader, APIKey
from starlette.status import HTTP_403_FORBIDDEN
from transformers import DistilBertTokenizerFast

from pydantic import BaseModel, Field
from typing import List
from geomodel.geomodel import GeoModel

BASE_MODEL = 'models/2021-12-03_model-distilbert-base-uncased_loss-wiki_exploded_geonames'
TOKEN_MODEL = 'distilbert-base-uncased'
API_VERSION = '1.0'
MODEL_VERSION = '1.0'
MAX_SEQ_LENGTH = 200

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
    description='''
    GEOFYR infers the geographic location of every kind of text.\
    GEOFYR is based on state-of-the-art Natural Language Processing \
    and smartly infers the to location of a text based on its semantic \
    and is not fooled by content which does not contribute \
    to the semantcs of a text.
    '''
)

class Coordinate(BaseModel):
    lat: float = Field(...,
                       example=52.5510,
                       title="Latitude",
                       description="Latitude of the predicted location."
                      )
    lon: float = Field(...,
                       example=13.3304,
                       title="Longitude",
                       description="Longitude of the predicted location."
                      )

class Meta(BaseModel):
    api_version: str = Field(...,
                        example='1.0',
                        description="Version number of the API."
                        )
    model_version: str = Field(...,
                        example='1.0',
                        description="Version number of the GEOFYR Neural Network."
                        )
    
class InputMetrics(BaseModel):
    input_tokens: int = Field(...,
                        example=300,
                        description="The number of input tokens which where detected."
                        )
    inference_tokens: int = Field(...,
                        example=200,
                        description="The number of input tokens which where used for inference."
                        )
    unused_tokens: int = Field(...,
                        example=100,
                        description=f"The number of input tokens which where ignored for inference.\
                        Only {MAX_SEQ_LENGTH} tokens are allowed for inference. \
                        Split your text beforehand into chunks of {MAX_SEQ_LENGTH} tokens for inference of larger texts."
                        )

class PointResponse(BaseModel):
    meta: Meta = Meta(
        api_version=API_VERSION, 
        model_version=MODEL_VERSION)
    coordinate: Coordinate
    input_metrics: InputMetrics
    
class GeoRequest(BaseModel):
    text: str = Field(...,
                      max_tokens=MAX_SEQ_LENGTH,
                      example="There, on Mount Horeb, God appeared to Moses as a burning bush, revealed to Moses his name ...",
                      title="Text",
                      description="Text to infer geograpic location from. Maximum 200 tokens. A token is everything between two whitespaces."
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

@app.post("/inference/point/", response_model = PointResponse)
async def inference_point(request: GeoRequest, api_key: APIKey = Depends(get_api_key)):
    '''
    Infers the geographic location of a given text as a single point.
    '''
    # Split text to tokens
    split_text = request.text.split(" ")
    
    # Slice tokens and calc token metrics
    input_tokens = len(split_text)
    inference_tokens = min(input_tokens, 200)
    inference_text = " ".join(split_text[:inference_tokens])
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

