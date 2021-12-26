from fastapi import FastAPI, Depends
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader, APIKey
from starlette.status import HTTP_403_FORBIDDEN
from starlette.responses import RedirectResponse

from transformers import DistilBertTokenizerFast
from sklearn.neighbors import KernelDensity

from model import GeoModel
from model import (
    Coordinate,
    InputMetrics,
    PointResponse,
    AreaResponse,
    GeoRequest)
from model import API_DESCRIPTION_STR
from model import MAX_SEQ_LENGTH, BASE_MODEL, TOKEN_MODEL, API_VERSION

kde = KernelDensity(
    kernel='gaussian',
    metric='haversine',
    bandwidth=0.006)

geomodel = GeoModel(
    model_string=BASE_MODEL,
    tokenizer_string=TOKEN_MODEL,
    tokenizer_class=DistilBertTokenizerFast,
    max_seq_length=MAX_SEQ_LENGTH,
    pdf=kde)

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
    # Inference
    geomodel.forward(request.text)
    point_coordinate = geomodel.predict_point()

    # Compose respond
    coordinate = Coordinate(lat=point_coordinate[0], lon=point_coordinate[1])
    input_metrics = InputMetrics(
        input_tokens=geomodel.input_tokens,
        inference_tokens=geomodel.inference_tokens,
        unused_tokens=geomodel.unused_tokens
    )
    return PointResponse(coordinate=coordinate, input_metrics=input_metrics)


@app.post("/inference/area", response_model=AreaResponse)
def inference_area(request: GeoRequest,
                   api_key: APIKey = Depends(validate_api_key)
                   ):
    '''
    Infers the geographic location of a given text as a single point.
    For security reasons this endpoint only accepts the text as encrypted SSL payload.
    '''
    # Inference
    geomodel.forward(request.text)
    polygon, bbox = geomodel.predict_area()

    # Compose respond
    polygon_list = [Coordinate(lat=coord[0], lon=coord[1]) for coord in polygon]
    bbox_list = [Coordinate(lat=coord[0], lon=coord[1]) for coord in bbox]
    input_metrics = InputMetrics(
        input_tokens=geomodel.input_tokens,
        inference_tokens=geomodel.inference_tokens,
        unused_tokens=geomodel.unused_tokens
    )
    return AreaResponse(area_polygon=polygon_list,
                        area_bbox=bbox_list,
                        input_metrics=input_metrics)
