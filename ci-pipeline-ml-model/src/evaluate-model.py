from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle


def evaluate_saved_model():
    data = load_wine()
    _,X_test,_,y_test = train_test_split(data.data,data.target,test_size=0.2,random_state=42)
    model = pickle.load(open("model.pkl",'rb'))
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test,predictions)
    return accuracy


if __name__ == "__main__":
    accuracy = evaluate_saved_model()
    print(f"Saved model accuracy :{accuracy}")

 