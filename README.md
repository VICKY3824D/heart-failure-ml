# Heart Failure Prediction - Flask Web App

Aplikasi web berbasis **Flask** untuk memprediksi risiko kematian pasien gagal jantung (`DEATH_EVENT`) menggunakan model *machine learning* (`heart_model.joblib`), lengkap dengan dashboard performa model.

## Prasyarat

- **Python 3.9+** (disarankan Python 3.10–3.12)
- **pip** (package manager Python)

## Instalasi & Menjalankan di Local

1. **Masuk ke direktori proyek**

   ```bash
   cd heart_failure
   ```

2. **Buat dan aktifkan virtual environment**

   - **Linux / macOS:**

     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

   - **Windows (PowerShell):**

     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```

   - **Windows (Command Prompt):**

     ```cmd
     python -m venv .venv
     .venv\Scripts\activate.bat
     ```

3. **Install dependensi**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   Paket yang diinstall dari `requirements.txt`: `flask`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, dan `joblib`.

4. **Jalankan server Flask**

   ```bash
   python heart_api.py
   ```

   *(Jika di Linux/macOS perintah `python` tidak tersedia, gunakan `python3 heart_api.py`.)*

5. **Buka di browser**

   Akses aplikasi melalui alamat berikut:

   - Halaman utama (redirect ke `/heart/`): `http://127.0.0.1:5000/`
   - Form prediksi pasien: `http://127.0.0.1:5000/heart/`
   - Dashboard evaluasi model: `http://127.0.0.1:5000/heart/dashboard/`

   Tekan `Ctrl + C` di terminal untuk menghentikan server.

## Endpoint Aplikasi

| Route | Method | Keterangan |
| :--- | :--- | :--- |
| `/` | `GET` | Redirect otomatis ke `/heart/` |
| `/heart/` | `GET` | Form input data klinis pasien |
| `/heart/result/` | `POST` | Menampilkan hasil prediksi dan probabilitas |
| `/heart/dashboard/` | `GET` | Menampilkan perbandingan model, metrik, dan grafik evaluasi |

## Struktur File

```text
heart_failure/
├── heart_api.py          # Aplikasi utama Flask & routing
├── heart_model.joblib    # Model terlatih
├── model_info.json       # Metadata model, metrik evaluasi, dan versi scikit-learn
├── requirements.txt      # Daftar dependensi Python
├── static/               # Aset CSS dan gambar grafik
└── templates/            # Template HTML (base, index, result, dashboard)
```

## Catatan

- **Versi `scikit-learn`**: Model pada `model_info.json` dilatih menggunakan `scikit-learn` versi `1.9.1`. Jika versi `scikit-learn` di environment Anda berbeda, aplikasi akan menampilkan peringatan di halaman web (atau gagal memuat `heart_model.joblib` jika formatnya tidak kompatibel).
- **Port sudah terpakai**: Secara default aplikasi berjalan di `127.0.0.1:5000`. Jika port `5000` sedang dipakai aplikasi lain, ubah baris `app.run()` pada `heart_api.py` menjadi misalnya `app.run(port=5001)`.
