import mlflow.sklearn
import pandas as pd
import numpy as np

def predict_data(model_path, data):
    """
    Fungsi untuk load model dan melakukan prediksi
    """
    print(f"Loading model from {model_path}...")
    

    model = mlflow.sklearn.load_model(model_path)
    

    prediction = model.predict(data)
    return prediction

if __name__ == "__main__":

    print("Simulasi Inference...")
    

    dummy_data = pd.DataFrame([{
        'ph': 7.0,
        'Hardness': 200.0,
        'Solids': 20000.0,
        'Chloramines': 7.0,
        'Sulfate': 300.0,
        'Conductivity': 400.0,
        'Organic_carbon': 15.0,
        'Trihalomethanes': 60.0,
        'Turbidity': 4.0
    }])
    

    
    print("Script inference siap digunakan untuk serving.")
