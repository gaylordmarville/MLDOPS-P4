from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from src.utils.helpers import timer_func
import logging

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s',
    force=True)


@timer_func
def find_model_and_params(X_train, y_train, X_test, y_test):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    # Create the parameter grid based on the results of random search
    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }

    l_cv = [3, 4, 5]

    final_best_params = dict()

    best_f1score = float("-inf")

    best_cv = None

    for cv in l_cv:

        # Create a based model
        clf = RandomForestClassifier()

        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid,
                                   cv=cv, n_jobs=-1, verbose=2,
                                   scoring="f1_macro")

        # Fit the grid search to the data
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_

        model = RandomForestClassifier(**best_params)

        model.fit(X_train, y_train)

        preds = inference(model, X_test)

        _, _, f1score = compute_model_metrics(y_test, preds)

        if f1score > best_f1score:
            best_f1score = f1score
            final_best_params = best_params
            best_cv = cv

    logging.info(f"Best cv: {best_cv}")
    logging.info(f"Best Random Forest classifier parameters: \
                 {final_best_params}")

    return RandomForestClassifier, final_best_params


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train, X_test, y_test, params_tuning=False):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    if params_tuning:
        model, best_parameters = find_model_and_params(X_train,
                                                       y_train,
                                                       X_test,
                                                       y_test)
        clf = model(**best_parameters)
        clf.fit(X_train, y_train)
    else:
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)

    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision,
    recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    y_pred = model.predict(X)

    return y_pred
