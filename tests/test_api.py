def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_create_and_get_analysis(client):
    res = client.post("/api/v1/analyses", json={"handle": "Example_User"})
    assert res.status_code == 201
    body = res.json()
    assert body["subject_handle"] == "example_user"
    assert body["status"] == "pending"

    res = client.get(f"/api/v1/analyses/{body['id']}")
    assert res.status_code == 200


def test_invalid_handle(client):
    res = client.post("/api/v1/analyses", json={"handle": "bad handle!"})
    assert res.status_code == 422


def test_analysis_not_found(client):
    assert client.get("/api/v1/analyses/9999").status_code == 404
