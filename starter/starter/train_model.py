# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from starter.starter.ml.data import process_data, load_data
from starter.starter.ml.model import train_model, compute_model_metrics
from starter.starter.ml.model import inference
import joblib
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s',
    force=True)

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

data_path = os.path.join(parent_dir, "data/clean_census.csv")
model_fd_path = os.path.join(parent_dir, "model/")

logging.info(f"data_path: {data_path}")

# data_path = "../data/clean_census.csv"

# Add code to load in the data.
data = load_data(data_path)

# Even if we use used the K-fold cross validation
# we keep a test set to evaluate the model later
train, test = train_test_split(data, test_size=0.20)

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

# Preprocess the entire dataset for k-fold validation.
X_train, y_train, encoder, lb = process_data(train,
                                             categorical_features=cat_features,
                                             label="salary", training=True)


X_test, y_test, _, _ = process_data(
                            test,
                            categorical_features=cat_features,
                            label="salary",
                            training=False,
                            encoder=encoder,
                            lb=lb)


# Train and save a model.
model = train_model(X_train, y_train, X_test, y_test, params_tuning=True)

# Model performance evaluation on categorical features
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

logging.info(f"Precision:{precision}, Recall:{recall}, F1 score:{fbeta}")

model_path = os.path.join(model_fd_path, "best_model.joblib")
logging.info(f"Dumping best model: {model_path}")
joblib.dump(model, open(model_path, "wb"))
