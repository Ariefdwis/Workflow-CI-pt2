import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():

    try:
        df = pd.read_csv('water_potability_clean.csv')
    except FileNotFoundError:
        print("Dataset tidak ditemukan")
        return

    X = df.drop('Potability', axis=1)
    y = df['Potability']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    with mlflow.start_run():
        clf = RandomForestClassifier(n_estimators=10)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {acc}")
        
   
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(clf, "model")

if __name__ == "__main__":
    main()