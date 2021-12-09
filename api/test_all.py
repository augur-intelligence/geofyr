from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Welcome to GEOFYR. Visit /docs for documentation <3"}

    
def test_inference_point():
    header = {"access_token" : "geo"}
    data = {"text" : "Berlin is a city in Germany."}
    response = client.post("/inference/point", json=data, headers=header)
    json = response.json()
    json.keys()
    assert response.status_code == 200
    assert "meta" in json.keys()
    assert "coordinate" in json.keys()
    assert "input_metrics" in json.keys()
    assert "lat" in json['coordinate'].keys()
    assert "lon" in json['coordinate'].keys()
    assert type(json['coordinate']['lat']) == float
    assert type(json['coordinate']['lon']) == float
    assert "api_version" in json['meta'].keys()
    assert "model_version" in json['meta'].keys()
    assert type(json['meta']['api_version']) == str
    assert type(json['meta']['model_version']) == str
    assert "input_tokens" in json['input_metrics'].keys()
    assert "inference_tokens" in json['input_metrics'].keys()
    assert "unused_tokens" in json['input_metrics'].keys()
    assert type(json['input_metrics']['input_tokens']) == int
    assert type(json['input_metrics']['inference_tokens']) == int
    assert type(json['input_metrics']['unused_tokens']) == int
