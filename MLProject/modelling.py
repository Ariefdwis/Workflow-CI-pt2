import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



DAGSHUB_USERNAME = "Ariefdwis"  
DAGSHUB_REPO_NAME = "Eksperimen_MSML_AriefDwiSeptian_pt2" 

def main():
    print("Memulai proses Training Model ")

    
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    mlflow.set_experiment("Eksperimen Water Potability")

    
    try:
        df = pd.read_csv('water_potability_clean.csv')
    except FileNotFoundError:
        print("File 'water_potability_clean.csv' tidak ditemukan di folder ini!")
        return

    X = df.drop('Potability', axis=1)
    y = df['Potability']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    with mlflow.start_run():
        print(" Sedang melakukan Hyperparameter Tuning...")
        
        
        rf = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        
     
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=1, verbose=1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Akurasi Terbaik: {acc:.4f}")
        print(f"Parameter Terbaik: {best_params}")

        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", acc)

        
        mlflow.sklearn.log_model(best_model, "model_random_forest")

        
        plt.figure(figsize=(6,5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Aktual')
        plt.xlabel('Prediksi')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png") 

        
        report = classification_report(y_test, y_pred)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt") 
        
        print("Selesai! Model dan Artefak telah terkirim ke DagsHub.")

if __name__ == "__main__":
    main()
