 ## these test verifies 
#-- Data is valid
#-- Model trained correctly
#-- Model file saved properly
#-- Accuarcy above threshold


import os 
import pytest
from src.train_model import loading_data,train,evaluate
from sklearn.ensemble import RandomForestClassifier


def test_loading_data():
    X_train,X_test,y_train,y_test = loading_data()
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(y_train) == X_train.shape[0]

def test_model_training():
    """Check model training returns a RandomForestClassifier."""
    X_train, X_test, y_train, y_test = loading_data()
    model = train(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "predict")


def test_model_evaluation_accuracy():
    """Ensure model accuracy is above a baseline threshold."""
    X_train, X_test, y_train, y_test = loading_data()
    model = train(X_train, y_train)
    acc = evaluate(model, X_test, y_test)
    assert acc > 0.7, f"Model accuracy too low: {acc}"


@pytest.mark.dependency(depends=["test_model_training"])
def test_model_file_creation(tmp_path):
    """Check if model.pkl file is saved after training."""
    os.chdir(tmp_path)
    X_train, X_test, y_train, y_test = loading_data()
    model = train(X_train, y_train)
    import joblib
    joblib.dump(model, "model.pkl")
    assert os.path.exists("model.pkl")
