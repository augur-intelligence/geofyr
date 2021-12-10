import docker
import requests
from timeit import default_timer as timer

PORT = 8080
client = docker.from_env()
image = client.images.build(path='./', tag='geoapi', quiet=False)

def test_container():
    container = client.containers.run(image=image[0], ports={8080:PORT}, detach=True)
    start_container = timer()
    while not container.status == 'running': container.reload()
    while not "Uvicorn running" in container.logs().decode("utf-8"): True
    stop_container = timer()
    start_request = timer()
    response = requests.get(f'http://127.0.0.1:{PORT}')
    stop_request = timer()
    diff_container = stop_container - start_container
    diff_request = stop_request - start_request
    print(f"Container start: {diff_container:.3} seconds.")
    print(f"Response: {diff_request:.3} seconds.")
    assert response.status_code == 200
    assert (diff_container) < 5
    assert (diff_request) < 0.1
    container.stop()
