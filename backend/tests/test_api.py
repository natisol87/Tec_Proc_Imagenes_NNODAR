"""The API: contract with the frontend.    pytest tests/test_api.py -v"""

import base64
import json

import numpy as np
from PIL import Image
from io import BytesIO

from helpers import disc_mask, figure_image, png, random_image


def decode(data_url: str) -> np.ndarray:
    return np.array(Image.open(BytesIO(base64.b64decode(data_url.split(",", 1)[1]))))


def post_step(client, name, files=None, **data):
    files = {k: (f"{k}.png", v, "image/png") for k, v in (files or {}).items()}
    return client.post(f"/api/steps/{name}", files=files, data=data)


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_frontend_is_served(client):
    response = client.get("/")
    assert response.status_code == 200 and "PASTICHE" in response.text


def test_steps_publish_their_factors(client):
    steps = client.get("/api/steps").json()
    assert len(steps) == 5
    margin = next(f for f in steps[0]["factors"] if f["name"] == "margin")["schema"]
    assert margin["minimum"] == 0.3 and margin["maximum"] == 2.0 and margin["title"]


def test_palettes(client):
    palettes = client.get("/api/palettes").json()
    assert len(palettes) >= 8 and all(len(p["zones"]) == 5 for p in palettes)


def layer(response, name: str) -> bytes:
    return base64.b64decode(response.json()["layers"][name].split(",", 1)[1])


def test_steps_are_in_the_silkscreen_order(client):
    names = [s["name"] for s in client.get("/api/steps").json()]
    assert names == ["frame", "ink", "background", "zones", "compose"]


def test_whole_chain_one_step_at_a_time(client):
    image = png(random_image(512, 512))  # step 1 needs a real face: it has its own tests

    r = post_step(client, "ink", {"image": image})
    assert r.status_code == 200, r.text
    edges = layer(r, "edges")

    r = post_step(client, "background", {"image": image, "edges": edges})
    assert r.status_code == 200, r.text
    mask = layer(r, "mask")

    r = post_step(client, "zones", {"image": image, "mask": mask, "edges": edges})
    assert r.status_code == 200, r.text
    labels = layer(r, "labels")

    r = post_step(client, "compose", {"labels": labels, "mask": mask, "edges": edges},
                  params=json.dumps({"rows": 1, "cols": 2, "palettes": ["marilyn", "red"],
                                     "custom": [{"skin": "#ff9900", "lips": "#990000"}, {}]}))
    assert r.status_code == 200, r.text
    assert decode(r.json()["layers"]["final"]).shape == (512, 1024, 3)


def test_face_landmarks_travel_as_json(client):
    face = {"x": 40, "y": 20, "width": 48, "height": 56, "score": 0.9,  # fields that the step does not use are ignored
            "eyes": [[54, 40], [74, 40]]}
    r = post_step(client, "zones", {"image": png(figure_image()), "mask": png(disc_mask())},
                  face=json.dumps(face))
    assert r.status_code == 200, r.text


def test_colors(client):
    colors = client.get("/api/colors").json()
    assert len(colors) >= 20 and all(c["hex"].startswith("#") and c["label"] for c in colors)


def test_invalid_custom_color_is_rejected(client):
    r = post_step(client, "compose", {"labels": png(disc_mask()), "mask": png(disc_mask())},
                  params=json.dumps({"custom": [{"skin": "rosa"}]}))
    assert r.status_code == 400 and r.json()["detail"]["code"] == "INVALID_PARAMETERS"


def test_missing_layer_is_a_400_in_spanish(client):
    r = post_step(client, "zones", {"image": png(figure_image())})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "MISSING_INPUT"
    assert "guardá" in r.json()["detail"]["message"]


def test_factor_out_of_range(client):
    r = post_step(client, "zones", {"image": png(figure_image()), "mask": png(disc_mask())},
                  params=json.dumps({"eye_size": 99}))
    assert r.status_code == 400
    assert "como máximo" in r.json()["detail"]["message"]


def test_no_face(client):
    flat = np.full((200, 200, 3), 128, np.uint8)
    r = post_step(client, "frame", {"photo": png(flat)})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "NO_FACE"


def test_not_an_image(client):
    r = client.post("/api/steps/frame", files={"photo": ("a.txt", b"hola", "text/plain")})
    assert r.status_code == 400 and r.json()["detail"]["code"] == "INVALID_FILE"


def test_layers_of_different_sizes(client):
    r = post_step(client, "zones", {"image": png(figure_image(128)), "mask": png(disc_mask(64))})
    assert r.status_code == 400 and r.json()["detail"]["code"] == "SIZE_MISMATCH"


def test_bad_url(client):
    r = client.post("/api/photo/from-url", json={"url": "ftp://algo"})
    assert r.status_code == 400 and r.json()["detail"]["code"] == "INVALID_URL"
