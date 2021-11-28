import requests
import json

payload = {"age": 30,
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
           "native-country": "United-States"}

r = requests.post("http://0.0.0.0:5000/inference", data=json.dumps(payload))

print(r, r.content)
