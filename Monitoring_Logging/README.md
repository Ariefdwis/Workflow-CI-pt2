# Monitoring & Logging (Prometheus + Grafana)

Folder ini berisi konfigurasi dan bukti implementasi sistem monitoring untuk proyek Machine Learning.

##  Struktur File
- **`prometheus_exporter.py`**: Script Python untuk menjalankan *dummy exporter* dan menghasilkan metriks tiruan.
- **`prometheus.yml`**: File konfigurasi untuk menghubungkan Prometheus dengan exporter.
- **`requirements.txt`**: Daftar library yang dibutuhkan (`prometheus-client`).
- **`bukti_...`**: Screenshot bukti implementasi (Prometheus UI, Grafana Dashboard, Alerting).

##  Cara Menjalankan

1. **Install Library:**
   Pastikan Python sudah terinstall, lalu jalankan:
   pip install -r requirements.txt
2. **Jalankan Exporter::**
    python prometheus_exporter.py
    Exporter akan berjalan di port 8000.
3. **Jalankan Prometheus & Grafana:**
    Pastikan Docker Desktop sudah menyala, lalu jalankan container Prometheus dan Grafana sesuai konfigurasi Docker Compose atau manual run.
4. **Akses Dashboard:**
    Prometheus: http://localhost:9090
    Grafana: http://localhost:3000
5.  **Save** 
    (`Ctrl+S`)
