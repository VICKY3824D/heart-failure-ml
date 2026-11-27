"""
heart_api.py
Dashboard Flask untuk prediksi kematian pasien gagal jantung (DEATH_EVENT).

Halaman:
  /heart/            -> form input data pasien
  /heart/result/     -> hasil prediksi (POST dari form)
  /heart/dashboard/  -> performa model, perbandingan model, dan grafik

Jalankan:
    python heart_api.py
lalu buka http://127.0.0.1:5000/heart/
"""
import json
import os

import joblib
import pandas as pd
import sklearn
from flask import Flask, redirect, render_template, request, url_for

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "heart_model.joblib")
INFO_FILE = os.path.join(BASE_DIR, "model_info.json")

app = Flask(__name__)

# ----------------------------------------------------------------------
# Definisi field form. Urutan FEATURE_NAMES HARUS sama dengan saat training
# (lihat FEATURES di create_model/heart_failure/save_model.py).
# min/max hanya batas kewajaran input, bukan rentang data training.
# ----------------------------------------------------------------------
FIELDS = [
    {"name": "age", "label": "Usia", "unit": "tahun", "type": "number", "min": 1, "max": 120, "step": "1", "example": 60},
    {"name": "anaemia", "label": "Anemia", "type": "binary", "options": [("0", "Tidak"), ("1", "Ya")]},
    {"name": "creatinine_phosphokinase", "label": "Creatinine Phosphokinase (CPK)", "unit": "mcg/L", "type": "number", "min": 1, "max": 20000, "step": "1", "example": 250},
    {"name": "diabetes", "label": "Diabetes", "type": "binary", "options": [("0", "Tidak"), ("1", "Ya")]},
    {"name": "ejection_fraction", "label": "Ejection Fraction", "unit": "%", "type": "number", "min": 5, "max": 90, "step": "1", "example": 38},
    {"name": "high_blood_pressure", "label": "Hipertensi", "type": "binary", "options": [("0", "Tidak"), ("1", "Ya")]},
    {"name": "platelets", "label": "Trombosit (Platelets)", "unit": "kiloplatelets/mL", "type": "number", "min": 10000, "max": 1000000, "step": "any", "example": 262000},
    {"name": "serum_creatinine", "label": "Serum Creatinine", "unit": "mg/dL", "type": "number", "min": 0.1, "max": 15, "step": "any", "example": 1.1},
    {"name": "serum_sodium", "label": "Serum Sodium", "unit": "mEq/L", "type": "number", "min": 100, "max": 160, "step": "1", "example": 137},
    {"name": "sex", "label": "Jenis Kelamin", "type": "binary", "options": [("1", "Laki-laki"), ("0", "Perempuan")]},
    {"name": "smoking", "label": "Merokok", "type": "binary", "options": [("0", "Tidak"), ("1", "Ya")]},
]
FEATURE_NAMES = [f["name"] for f in FIELDS]

# ----------------------------------------------------------------------
# Muat model & info sekali saat aplikasi mulai (tidak per request)
# ----------------------------------------------------------------------
MODEL = None
INFO = None
LOAD_ERROR = None
try:
    MODEL = joblib.load(MODEL_FILE)
except Exception as e:  # file belum ada / versi sklearn tidak cocok
    LOAD_ERROR = (f"Model tidak dapat dimuat ({type(e).__name__}). Jalankan ulang "
                  "create_model/heart_failure/save_model.py di environment yang sama.")
try:
    with open(INFO_FILE, encoding="utf-8") as f:
        INFO = json.load(f)
except Exception:
    INFO = None


def version_warning():
    """Peringatan jika versi scikit-learn saat training berbeda dengan saat ini."""
    if INFO and INFO.get("sklearn_version") != sklearn.__version__:
        return (f"Model dilatih dengan scikit-learn {INFO['sklearn_version']}, sedangkan aplikasi "
                f"memakai {sklearn.__version__}. Latih ulang model agar hasil konsisten.")
    return None


def parse_form(form):
    """Validasi input form. Mengembalikan (values, errors)."""
    values, errors = {}, {}
    for f in FIELDS:
        raw = (form.get(f["name"]) or "").strip()
        if raw == "":
            errors[f["name"]] = "Wajib diisi."
            continue
        if f["type"] == "binary":
            if raw not in [o[0] for o in f["options"]]:
                errors[f["name"]] = "Pilihan tidak valid."
            else:
                values[f["name"]] = float(raw)
            continue
        try:
            v = float(raw)
        except ValueError:
            errors[f["name"]] = "Harus berupa angka."
            continue
        if not (f["min"] <= v <= f["max"]):
            errors[f["name"]] = f"Di luar rentang wajar ({f['min']} - {f['max']})."
        else:
            values[f["name"]] = v
    return values, errors


@app.route("/")
def home():
    return redirect(url_for("index"))


@app.route("/heart/")
def index():
    return render_template("index.html", fields=FIELDS, form={}, errors={},
                           load_error=LOAD_ERROR, warning=version_warning(), active="predict")


@app.route("/heart/result/", methods=["POST"])
def prediction_result():
    values, errors = parse_form(request.form)
    if errors or MODEL is None:
        return render_template("index.html", fields=FIELDS, form=request.form, errors=errors,
                               load_error=LOAD_ERROR, warning=version_warning(), active="predict"), 400

    # DataFrame dengan urutan kolom yang sama seperti saat training
    df_input = pd.DataFrame([[values[n] for n in FEATURE_NAMES]], columns=FEATURE_NAMES)
    pred = int(MODEL.predict(df_input)[0])
    prob_dead = float(MODEL.predict_proba(df_input)[0][1])

    status = "Deceased (Meninggal)" if pred == 1 else "Survived (Bertahan hidup)"
    rows = []
    for f in FIELDS:
        raw = request.form.get(f["name"], "").strip()
        shown = dict(f["options"]).get(raw, raw) if f["type"] == "binary" else raw
        rows.append((f["label"] + (f" ({f['unit']})" if f.get("unit") else ""), shown))

    return render_template("result.html", rows=rows, pred=pred, status=status,
                           prob_dead=round(prob_dead * 100, 1), prob_survive=round((1 - prob_dead) * 100, 1),
                           model_name=INFO["best_model"] if INFO else "-",
                           warning=version_warning(), active="predict")


@app.route("/heart/dashboard/")
def dashboard():
    return render_template("dashboard.html", info=INFO, load_error=LOAD_ERROR,
                           warning=version_warning(), active="dashboard")


if __name__ == "__main__":
    # host='127.0.0.1', port=5000 (default). Ubah jika port sudah terpakai.
    app.run()
