from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_get_path():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"fetch": "Welcome to salary prediction API"}


def test_post_inference_query_less():
    r = client.post("/inference",
                    json={"age": 30,
                          "workclass": "Private",
                          "fnlwgt": 13769,
                          "education": "Some-college",
                          "education-num": 10,
                          "marital-status": "Married-civ-spouse",
                          "occupation": "Machine-op-inspct",
                          "relationship": "Husband",
                          "race": "Amer-Indian-Eskimo",
                          "sex": "Male",
                          "capital-gain": 0,
                          "capital-loss": 0,
                          "hours-per-week": 30,
                          "native-country": "United-States"
                          },
                    )
    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}


def test_post_inference_query_more():
    r = client.post("/inference",
                    json={"age": 57,
                          "workclass": "Federal-gov",
                          "fnlwgt": 337895,
                          "education": "Bachelors",
                          "education-num": 13,
                          "marital-status": "Married-civ-spouse",
                          "occupation": "Prof-specialty",
                          "relationship": "Husband",
                          "race": "Black",
                          "sex": "Male",
                          "capital-gain": 0,
                          "capital-loss": 0,
                          "hours-per-week": 40,
                          "native-country": "United-States"
                          },
                    )
    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}
