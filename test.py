import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://44.202.56.219:5000")
mlflow.set_experiment("test")

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

with mlflow.start_run():
    mlflow.log_param("n_estimators", 10)
    mlflow.log_param("random_state", 42)

    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(model, "test_model")

    print("Expérience enregistrée sur le serveur MLflow")
