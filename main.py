from fastapi import FastAPI
from pydantic import BaseModel, Field
import os
import logging
import pandas as pd
import joblib
from src.ml.model import inference
from src.ml.data import process_data
import uvicorn


parent_dir = os.path.dirname(os.path.realpath(__file__))

model_path = os.path.join(parent_dir, "model/best_model.joblib")
encoder_path = os.path.join(parent_dir, "data/encoder.joblib")
label_binarizer_path = os.path.join(parent_dir, "data/label_binarizer.joblib")
scaler_path = os.path.join(parent_dir, "data/scaler.joblib")

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s',
    force=True)

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()


class Input(BaseModel):

    age: int = Field(default=30)
    workclass: str = Field(default="Private")
    fnlwgt: int = Field(default=13769)
    education: str = Field(default="Some-college",
                           alias='education')
    education_num: int = Field(default=10, alias='education-num')
    marital_status: str = Field(default="Married-civ-spouse",
                                alias='marital-status')
    occupation: str = Field(default="Machine-op-inspct")
    relationship: str = Field(default="Husband")
    race: str = Field(default="Amer-Indian-Eskimo")
    sex: str = Field(default="Male")
    capital_gain: int = Field(default=0, alias="capital-gain")
    capital_loss: int = Field(default=0, alias='capital-loss')
    hours_per_week: int = Field(default=30, alias='hours-per-week')
    native_country: str = Field(default="United-States",
                                alias='native-country')


@app.get("/")
async def get_items():
    return {"fetch": "Welcome to salary prediction API"}


@app.post("/inference")
async def get_inference(body: Input):

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    d_input = dict()

    model = joblib.load(open(model_path, "rb"))
    encoder = joblib.load(open(encoder_path, "rb"))
    lb = joblib.load(open(label_binarizer_path, "rb"))
    scaler = joblib.load(open(scaler_path, "rb"))

    for k, v in body.dict(by_alias=True).items():
        d_input[k] = [v]
    input_ftrs = pd.DataFrame.from_dict(d_input)
    X_infer, _, _, _, _ = process_data(input_ftrs,
                                       categorical_features=cat_features,
                                       training=False,
                                       encoder=encoder,
                                       lb=lb,
                                       scaler=scaler)

    preds = inference(model, X_infer)
    return {"prediction": ">50K" if preds[0] else "<=50K"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=4)
