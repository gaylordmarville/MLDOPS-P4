# Add the necessary imports for the starter code.
from src.ml.data import process_data, load_data
from src.ml.model import perf_on_slices, train_model, compute_model_metrics
from src.ml.model import inference
import joblib
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True)

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

train_data_path = os.path.join(parent_dir, "data/adult_cleaned.data")
test_data_path = os.path.join(parent_dir, "data/adult_cleaned.test")
model_path = os.path.join(parent_dir, "model/best_model.joblib")
encoder_path = os.path.join(parent_dir, "data/encoder.joblib")
label_binarizer_path = os.path.join(parent_dir, "data/label_binarizer.joblib")
scaler_path = os.path.join(parent_dir, "data/scaler.joblib")


def task(training_mode):

    # Even if we use used the K-fold cross validation
    # we keep a test set to evaluate the model later
    # preserving the split made by UCI using MLC++
    train, test = load_data(train_data_path, test_data_path)

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
    logging.info("Dumping encoder complete.")

    logging.info("Dumping label binarizer...")
    joblib.dump(lb, open(label_binarizer_path, "wb"))
    logging.info("Dumping label binarizer complete.")

    logging.info("Dumping scaler...")
    joblib.dump(scaler, open(scaler_path, "wb"))
    logging.info("Dumping scaler complete.")

    X_test, y_test, _, _, _ = process_data(
                                test,
                                categorical_features=cat_features,
                                label="salary",
                                training=False,
                                encoder=encoder,
                                lb=lb,
                                scaler=scaler)

    if training_mode:
        # Train and save a model.
        logging.info("Starting model training...")
        model = train_model(X_train, y_train, X_test, y_test,
                            params_tuning=True)
        logging.info("Model training complete.")

    else:
        # Infer on dumped model
        logging.info("Loading model...")
        model = joblib.load(model_path)
        logging.info("Loading model complete.")

    # TODO: Write a function that computes performance on model slices.
    feature = "education"
    logging.info(f"Compute performance on {feature} slices...")
    perf_on_slices(feature, process_data, test, cat_features, encoder, lb,
                   scaler, model)
    logging.info("Performance on education slices complete.")

    # Model performance evaluation on categorical features
    logging.info("Starting model inference...")
    preds = inference(model, X_test)
    logging.info("Model inference complete.")

    logging.info("Compute model metrics...")
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    logging.info("Model metrics complete...")
    logging.info(f"Precision:{precision}, Recall:{recall}, F1 score:{fbeta}")

    # Model dumping
    logging.info("Dumping best model...")
    joblib.dump(model, open(model_path, "wb"))
    logging.info("Dumping best model complete.")
    logging.info(f"Model file path: {model_path}")
