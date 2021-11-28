import logging
from src.ml.data import load_data, process_data
import numpy as np
import os

logging.basicConfig(
    filename='./tests/logs/tests.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True)

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

train_data_path = os.path.join(parent_dir, "data/adult_cleaned.data")


DATA = None


def test_import():
    '''
    test data import - this example is completed for you
    to assist with the other test functions
    '''
    global DATA
    try:
        DATA = load_data(train_data_path)
        logging.info("Testing load: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert DATA.shape[0] > 0
        assert DATA.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data:\
             The file doesn't appear to have rows and columns")
        raise err


def test_standardization():
    try:
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

        X_train, y_train, _, _, _ = process_data(
                                 DATA,
                                 categorical_features=cat_features,
                                 label="salary",
                                 training=True)

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
        logging.error("Testing process_data:\
                      The data are not correctly preprocessed")
        raise err
