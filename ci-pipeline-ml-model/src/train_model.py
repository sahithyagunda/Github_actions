from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle


def loading_data():
    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train(X_train, y_train):
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = loading_data()
    model = train(X_train, y_train)
    accuracy = evaluate(model, X_test, y_test)
    pickle.dump(model, open('model.pkl', 'wb'))
    print(f"Model trained successfully with accuracy {accuracy:2f}")
