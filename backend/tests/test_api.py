"""The API: contract with the frontend.    pytest tests/test_api.py -v"""


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_frontend_is_served(client):
    response = client.get("/")
    assert response.status_code == 200 and "PASTICHE" in response.text
