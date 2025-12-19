import time
import random
import requests
import pandas as pd
from prometheus_client import start_http_server, Gauge, Counter, Summary

# --- KONFIGURASI PROMETHEUS  ---
# 1. Counter: Jumlah Request
REQUEST_COUNT = Counter('request_count', 'Total Request yang masuk')
# 2. Gauge: Akurasi Model (Simulasi)
MODEL_ACCURACY = Gauge('model_accuracy', 'Akurasi model saat ini')
# 3. Summary: Latency/Waktu Proses
REQUEST_LATENCY = Summary('request_latency_seconds', 'Waktu proses request')
# 4. Gauge: Penggunaan Memory (Simulasi)
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Penggunaan memori aplikasi')
# 5. Gauge: CPU Usage (Simulasi)
CPU_USAGE = Gauge('cpu_usage_percent', 'Penggunaan CPU aplikasi')
# 6. Counter: Jumlah Prediksi Potable (1)
PRED_POTABLE_COUNT = Counter('pred_potable_count', 'Jumlah prediksi air layak minum')
# 7. Counter: Jumlah Prediksi Not Potable (0)
PRED_NOT_POTABLE_COUNT = Counter('pred_not_potable_count', 'Jumlah prediksi air tidak layak')
# 8. Gauge: Nilai Input pH Rata-rata
AVG_PH_INPUT = Gauge('avg_ph_input', 'Rata-rata input pH air')
# 9. Counter: Jumlah Error/Fail
ERROR_COUNT = Counter('error_count', 'Jumlah request gagal')
# 10. Gauge: Suhu Server (Simulasi)
SERVER_TEMP = Gauge('server_temp_celsius', 'Suhu server estimasi')

# --- DATASET DUMMY UNTUK REQUEST ---
def get_dummy_data():
    # Data acak menyerupai struktur dataset asli
    return {
        "ph": random.uniform(0, 14),
        "Hardness": random.uniform(100, 300),
        "Solids": random.uniform(10000, 50000),
        "Chloramines": random.uniform(4, 10),
        "Sulfate": random.uniform(200, 500),
        "Conductivity": random.uniform(300, 700),
        "Organic_carbon": random.uniform(10, 30),
        "Trihalomethanes": random.uniform(50, 100),
        "Turbidity": random.uniform(2, 6)
    }

def main():
    # Jalankan server Prometheus di port 8000
    print("Prometheus Exporter berjalan di port 8000...")
    start_http_server(8000)
    
    # URL Model 
    model_url = "http://localhost:5000/invocations" 
    
    
    while True:
        data = get_dummy_data()
        
        # Update Metriks Input
        AVG_PH_INPUT.set(data['ph'])
        SERVER_TEMP.set(random.uniform(40, 70))
        MEMORY_USAGE.set(random.uniform(200, 512) * 1024 * 1024)
        CPU_USAGE.set(random.uniform(10, 80))

        start_time = time.time()
        try:
            # Kirim request ke model (Simulasi format MLflow input)
            payload = {"dataframe_records": [data]}
            
            # Request ke model (Uncomment baris di bawah jika model docker sudah jalan)
            # response = requests.post(model_url, json=payload)
            # result = response.json()
            
            # --- SIMULASI HASIL ---
            prediction = random.choice([0, 1]) 
            REQUEST_COUNT.inc()
            
            if prediction == 1:
                PRED_POTABLE_COUNT.inc()
            else:
                PRED_NOT_POTABLE_COUNT.inc()

            MODEL_ACCURACY.set(random.uniform(0.65, 0.95)) # Simulasi akurasi fluktuatif

        except Exception as e:
            ERROR_COUNT.inc()
            print(f"Error: {e}")
        
        # Hitung Latency
        latency = time.time() - start_time
        REQUEST_LATENCY.observe(latency)
        
        print(f"Mengirim data... Latency: {latency:.4f}s")
        time.sleep(2) # Kirim data tiap 2 detik

if __name__ == "__main__":
    main()