import torch
from pydantic import BaseModel, Field
from typing import List
from scipy.spatial import ConvexHull
import numpy as np

BASE_MODEL = 'model/checkpoints-2021-12-22_model-distilbert-base-uncased_loss-wiki-utf-exploded-links-model-2.pt'
TOKEN_MODEL = 'distilbert-base-uncased'
API_VERSION = '1.0'
MODEL_VERSION = '1.0'
MAX_SEQ_LENGTH = 200


class GeoModel():
    def __init__(self, model_string, tokenizer_string, tokenizer_class, max_seq_length, pdf):
        self.model_string = model_string
        self.tokenizer_string = tokenizer_string
        self.tokenizer_class = tokenizer_class
        self.tokenizer = self.tokenizer_class.from_pretrained(self.tokenizer_string)
        self.model = torch.load(
            self.model_string,
            map_location=torch.device('cpu'))
        self.max_seq_length = max_seq_length
        self.pdf = pdf
        self.current_text = ''

    def forward(self, text):
        self.current_text = text
        # Split text to tokens
        self.split_text = self.current_text.split(" ")
        # Slice tokens and calc token metrics
        self.input_tokens = len(self.split_text)
        self.inference_tokens = min(self.input_tokens, self.max_seq_length)
        self.inference_text = str(" ").join(
            self.split_text[:self.inference_tokens])
        self.unused_tokens = len(self.split_text[self.inference_tokens:])
        # Tokenize
        tokenized_text = self.tokenizer(
            self.inference_text,
            truncation=True,
            padding=True,
            return_tensors='pt',
            max_length=self.max_seq_length)
        # Set to eval mode andd do inference
        self.model.eval()
        with torch.no_grad():
            self.output = self.model.forward(
                input_ids=tokenized_text['input_ids'],
                attention_mask=tokenized_text['attention_mask'],
                output_hidden_states=True)

    def predict_point(self):
        return self.output['logits'].numpy().squeeze()

    def predict_area(self):
        # Get pooled output from last hidden state
        last_hidden_state = self.output['hidden_states'][-1]
        hidden_state = last_hidden_state
        pooled_output = hidden_state[:, 0]
        rand_locs = []
        pooled_output_dim = pooled_output.shape[1]
        with torch.no_grad():
            self.model.eval()
            # Mask every dim of pooled output before predicition
            # Gather all outputs from masked predicitons
            for i in range(pooled_output_dim):
                ones = torch.ones_like(pooled_output)
                ones[0, i] = 0
                masked_output = ones * pooled_output
                pre_clf_output = self.model.pre_classifier(masked_output)
                relu_output = torch.nn.ReLU()(pre_clf_output)
                rand_loc = (self
                            .model
                            .classifier(relu_output)
                            .cpu()
                            .detach()
                            .numpy()
                            .squeeze())
                rand_locs.append(rand_loc)

        # Fit PDF on masked output and sample.
        # Make sure dist metric is haversine
        self.pdf.fit(np.radians(rand_locs))
        kde_sample = np.rad2deg(self.pdf.sample(1000))
        # Get convex hull and bounding box of sample
        kde_area_hull = ConvexHull(kde_sample)
        polygon = kde_area_hull.points[kde_area_hull.vertices]
        bbox = [[polygon[:, 0].min(), polygon[:, 1].min()],
                [polygon[:, 0].max(), polygon[:, 1].max()]]
        return polygon.tolist(), bbox


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


class MetaInfo(BaseModel):
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
    meta: MetaInfo = MetaInfo(
        api_version=API_VERSION, 
        model_version=MODEL_VERSION)
    coordinate: Coordinate = Field(...,
                                   description="The predicted coordinate of the texts location."
                              )
    input_metrics: InputMetrics


class AreaResponse(BaseModel):
    meta: MetaInfo = MetaInfo(
        api_version=API_VERSION, 
        model_version=MODEL_VERSION)
    area_polygon: List[Coordinate] = Field(...,
                              description="Array of coordinates, representing the polygon of the convex hull of the 90 percent confidence area, calibrated on the test data."
                              )
    area_bbox: List[Coordinate] = Field(...,
                              description="Array of coordinates, representing the bounding box of the polygon."
                              )
    input_metrics: InputMetrics


class GeoRequest(BaseModel):
    text: str = Field(...,
                      max_tokens=MAX_SEQ_LENGTH,
                      example="There, on Mount Horeb, God appeared to Moses as a burning bush, revealed to Moses his name ...",
                      title="Text",
                      description="Text to infer geograpic location from. Maximum 200 tokens. A token is everything between two whitespaces. Make sure the string is web safe and all characters are escpaed properly."
                      )

API_DESCRIPTION_STR = open("text/api_desc").read()