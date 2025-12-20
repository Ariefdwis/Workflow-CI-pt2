import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
import dagshub
import os

def main():

    dagshub_token = os.environ.get("DAGSHUB_USER_TOKEN")
    if dagshub_token:
        print("Authenticating with DagsHub Token...")
        dagshub.auth.add_app_token(dagshub_token)
    

    dagshub.init(repo_owner='Ariefdwis', repo_name='Eksperimen_MSML_AriefDwiSeptian_pt2', mlflow=True)
    

    mlflow.set_tracking_uri("https://dagshub.com/Ariefdwis/Eksperimen_MSML_AriefDwiSeptian_pt2.mlflow")
    mlflow.set_experiment("CI_CD_Retraining_Experiment")

    print("MLflow Tracking URI:", mlflow.get_tracking_uri())


    df = pd.read_csv('water_potability_clean.csv')

    df.fillna(df.mean(), inplace=True) 

    X = df.drop('Potability', axis=1)
    y = df['Potability']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    mlflow.autolog()

    with mlflow.start_run():
        print("Mulai Training Model di CI/CD...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        print(f"Model Accuracy: {accuracy}")
        
   
        mlflow.log_metric("accuracy_manual", accuracy)

    print("Training Selesai! Data terkirim ke DagsHub.")

if __name__ == "__main__":
    main()