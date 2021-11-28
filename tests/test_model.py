import logging
from src.ml.model import compute_model_metrics
import numpy as np

logging.basicConfig(
    filename='./tests/logs/tests.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True)


def test_compute_model_metrics():
    try:
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_preds = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        precision, recall, fbeta = compute_model_metrics(y, y_preds)

        assert [precision, recall, fbeta] == [0.5]*3

        logging.info("Testing compute_model_metrics: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing compute_model_metrics: \
                      The metrics evaluation failed")
        raise err
