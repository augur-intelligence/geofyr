import torch
from pydantic import BaseModel, Field
from typing import List

BASE_MODEL = 'model/model.pt'
TOKEN_MODEL = 'distilbert-base-uncased'
API_VERSION = '1.0'
MODEL_VERSION = '1.0'
MAX_SEQ_LENGTH = 200


class GeoModel():
    def __init__(self, model_string, tokenizer_string, tokenizer_class, max_seq_length):
        self.model_string = model_string
        self.tokenizer_string = tokenizer_string
        self.tokenizer_class = tokenizer_class
        self.tokenizer = self.tokenizer_class.from_pretrained(self.tokenizer_string)
        self.model = torch.load(
            self.model_string,
            map_location=torch.device('cpu'))
        self.max_seq_lengt = max_seq_length

    def forward(self, text):
        self.current_text = text
        tokenized_text = self.tokenizer(
            self.current_text,
            truncation=True,
            padding=True,
            return_tensors='pt',
            max_length=self.max_seq_lengt)

        self.model.eval()
        with torch.no_grad():
            self.output = self.model.forward(
                input_ids=tokenized_text['input_ids'],
                attention_mask=tokenized_text['attention_mask'],
                output_hidden_states=True)

    def predict_point(self):
        return self.output['logits'].numpy().squeeze()

    def predict_areas(self, num_layers):
        return self.output['logits'].numpy().squeeze()


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
                      description="Text to infer geograpic location from. Maximum 200 tokens. A token is everything between two whitespaces. Make sure the string is web safe and all characters are escpaed properly."
                      )


API_DESCRIPTION_STR = """
    GEOFYR infers the geographic location of every kind of text.
    GEOFYR is based on state-of-the-art Natural Language Processing and smartly infers the to location of the content of a text based on its meaning and semamntic. GEOFYR is able to process Literature, News, Historical text and everything else. If there is any clue where the action of a text occurs, GEOFYR will find it.
"""