import logging
from starter.starter.ml.data import load_data, process_data
from starter.starter.ml.model import compute_model_metrics
import numpy as np

logging.basicConfig(
    filename='./tests/logs/tests.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s',
    force=True)

def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        data = load_data("./starter/data/clean_census.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_standardization():
    try:
        data = load_data("./starter/data/clean_census.csv")

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

        X_train, y_train, _, _ = process_data(data,
                                              categorical_features=cat_features,
                                              label="salary", training=True)

        
        mean_cat = np.mean(X_train, axis=0)
        assert np.all(mean_cat < 1e-10) and np.all(mean_cat > -1e-10)
        assert np.all(np.std(X_train, axis=0))

        # Test we have the right number of features
        assert X_train.shape[1] == 108

        # Test label binarization
        assert len(np.unique(y_train)) == 2
        assert np.all(y_train >= 0) and np.all(y_train <= 1)
        assert type(y_train[0]) == np.int64

        logging.info("Testing process_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing process_data: The data are not correctly preprocessed")
        raise err


def test_compute_model_metrics():
    try:
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_preds = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        precision, recall, fbeta = compute_model_metrics(y, y_preds)

        assert [precision, recall, fbeta] == [0.5]*3

        logging.info("Testing compute_model_metrics: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing compute_model_metrics: The metrics evaluation failed")
        raise err