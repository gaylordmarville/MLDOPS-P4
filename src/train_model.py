# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from src.ml.data import process_data, load_data, load_data_2
from src.ml.model import train_model, compute_model_metrics
from src.ml.model import inference
import joblib
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s',
    force=True)

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

model_path = os.path.join(parent_dir, "model/best_model.joblib")
encoder_path = os.path.join(parent_dir, "data/encoder.joblib")
label_binarizer_path = os.path.join(parent_dir, "data/label_binarizer.joblib")
scaler_path = os.path.join(parent_dir, "data/scaler.joblib")

# Even if we use used the K-fold cross validation
# we keep a test set to evaluate the model later
# preserving the split made by UCI using MLC++
train, test = load_data()

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
X_train, y_train, encoder, lb, scaler = process_data(
                                        train,
                                        categorical_features=cat_features,
                                        label="salary", training=True)


logging.info("Dumping encoder...")
joblib.dump(encoder, open(encoder_path, "wb"))

logging.info("Dumping label binarizer...")
joblib.dump(lb, open(label_binarizer_path, "wb"))

logging.info("Dumping scaler...")
joblib.dump(scaler, open(scaler_path, "wb"))

X_test, y_test, _, _, _ = process_data(
                            test,
                            categorical_features=cat_features,
                            label="salary",
                            training=False,
                            encoder=encoder,
                            lb=lb,
                            scaler=scaler)


# Train and save a model.
logging.info("Starting model training...")
model = train_model(X_train, y_train, X_test, y_test, params_tuning=True)
logging.info("Model training complete")

# Model performance evaluation on categorical features
logging.info("Starting model inference...")
preds = inference(model, X_test)
logging.info("Model training complete")

logging.info("Compute model metrics...")
precision, recall, fbeta = compute_model_metrics(y_test, preds)
logging.info("Model metrics complete...")
logging.info(f"Precision:{precision}, Recall:{recall}, F1 score:{fbeta}")

# Model dumping
logging.info(f"Dumping best model: {model_path}")
joblib.dump(model, open(model_path, "wb"))
