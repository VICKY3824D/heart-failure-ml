"""
save_model.py
Pembuatan model prediksi kematian pasien gagal jantung (DEATH_EVENT).

Alur sama seperti notebook UTS:
  1. Import pustaka
  2. Load dataset
  3. Exploratory data analysis (EDA)
  4. Data preprocessing (hold-out 70:30 + StandardScaler)
  5. Training dan perbandingan 6 model
  6. Evaluasi mendalam (classification report, confusion matrix, ROC-AUC)
  7. Feature importance
  8. Hyperparameter tuning Random Forest (opsional: --tune)
  9. Simpan model terbaik (joblib) + info/grafik untuk dashboard Flask

Cara pakai (dari folder ini):
    python save_model.py
    python save_model.py --tune                       # tambah GridSearchCV Random Forest
    python save_model.py --model "Logistic Regression"  # paksa pilih model tertentu
"""
import argparse
import json
import os
import shutil
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")  # supaya bisa jalan tanpa layar/GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", ".."))
OUT_DIR = os.path.join(BASE_DIR, "output")
IMG_DIR = os.path.join(OUT_DIR, "img")

# Fitur input model (urutan HARUS sama dengan yang dipakai di aplikasi Flask).
# 'time' tidak dipakai karena lama follow-up baru diketahui setelah pasien diamati,
# sehingga tidak tersedia saat prediksi dilakukan.
FEATURES = ["age", "anaemia", "creatinine_phosphokinase", "diabetes",
            "ejection_fraction", "high_blood_pressure", "platelets",
            "serum_creatinine", "serum_sodium", "sex", "smoking"]
TARGET = "DEATH_EVENT"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(BASE_DIR, "heart_failure_clinical_records_dataset.csv"),
                   help="path file CSV dataset")
    p.add_argument("--tune", action="store_true", help="jalankan GridSearchCV untuk Random Forest")
    p.add_argument("--model", default=None, help="paksa memilih model tertentu (nama sesuai tabel)")
    return p.parse_args()


def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, name), dpi=130)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(IMG_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Load dataset
    # ------------------------------------------------------------------
    print("=" * 60)
    print("2. LOAD DATASET")
    print("=" * 60)
    if not os.path.exists(args.data):
        raise SystemExit(f"File dataset tidak ditemukan: {args.data}\n"
                         "Letakkan heart_failure_clinical_records_dataset.csv di folder ini "
                         "atau gunakan --data <path>.")
    df = pd.read_csv(args.data)
    print(df.head(10))
    print("Ukuran dataset:", df.shape)

    # ------------------------------------------------------------------
    # 3. EDA
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("3. EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(df.describe().T)
    print("\nMissing value per kolom:")
    print(df.isnull().sum())
    class_counts = df[TARGET].value_counts().sort_index()
    print("\nDistribusi target:")
    print(class_counts)

    class_counts.plot(kind="pie", autopct="%.2f%%", labels=["Survived (0)", "Deceased (1)"],
                      colors=["#2b6cb0", "#dd6b20"], figsize=(5, 5))
    plt.ylabel("")
    plt.title("Distribusi DEATH_EVENT")
    save_fig("class_dist.png")

    # ------------------------------------------------------------------
    # 4. Data preprocessing
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("4. DATA PREPROCESSING")
    print("=" * 60)
    # Semua fitur sudah numerik dan tidak ada missing value, sehingga tidak perlu
    # imputasi maupun encoding. Scaling dilakukan di dalam Pipeline (di-fit hanya
    # pada data training) supaya scaler ikut tersimpan bersama model.
    df = df.dropna()
    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print("Jumlah data training:", X_train.shape)
    print("Jumlah data testing :", X_test.shape)

    # ------------------------------------------------------------------
    # 5. Training dan perbandingan model
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("5. TRAINING DAN PERBANDINGAN MODEL")
    print("=" * 60)
    classifiers = {
        "Logistic Regression": LogisticRegression(),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=10),
        "Decision Tree": DecisionTreeClassifier(criterion="entropy", random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42),
    }
    models = {name: Pipeline([("scaler", StandardScaler()), ("clf", clf)])
              for name, clf in classifiers.items()}

    # ------------------------------------------------------------------
    # 8 (opsional). Hyperparameter tuning Random Forest
    # ------------------------------------------------------------------
    if args.tune:
        print("Menjalankan GridSearchCV untuk Random Forest ...")
        param_grid = {
            "clf__n_estimators": [50, 100, 150],
            "clf__max_depth": [None, 5, 10],
            "clf__min_samples_split": [2, 5, 10],
            "clf__criterion": ["gini", "entropy"],
        }
        grid = GridSearchCV(Pipeline([("scaler", StandardScaler()),
                                      ("clf", RandomForestClassifier(random_state=42))]),
                            param_grid=param_grid, cv=5, scoring="f1_macro", n_jobs=-1)
        grid.fit(X_train, y_train)
        print("Parameter terbaik:", grid.best_params_)
        models["Random Forest (Tuned)"] = grid.best_estimator_

    rows, probas = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)  # (best_estimator_ sudah fit, aman di-fit ulang)
        y_pred = model.predict(X_test)
        probas[name] = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "AUC": roc_auc_score(y_test, probas[name]),
            "Recall_Deceased": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        })
    comparison = pd.DataFrame(rows).sort_values(by=["F1", "Accuracy"], ascending=False).reset_index(drop=True)
    print(comparison.round(4).to_string())

    best_name = args.model if args.model else comparison.loc[0, "Model"]
    if best_name not in models:
        raise SystemExit(f"Model '{best_name}' tidak ada. Pilihan: {list(models)}")
    best_model = models[best_name]
    print(f"\n>> Model yang disimpan: {best_name}")

    # ------------------------------------------------------------------
    # 6. Evaluasi mendalam
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"6. EVALUASI MENDALAM ({best_name})")
    print("=" * 60)
    y_pred_best = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_best, target_names=["Survived (0)", "Deceased (1)"], zero_division=0))
    cm = confusion_matrix(y_test, y_pred_best)
    print("Confusion matrix:\n", cm)

    # grafik confusion matrix
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Survived", "Deceased"])
    ax.set_yticks([0, 1], ["Survived", "Deceased"])
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title(f"Confusion Matrix\n{best_name}")
    save_fig("confusion.png")

    # grafik ROC semua model
    plt.figure(figsize=(7, 5.5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Kurva ROC")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    save_fig("roc.png")

    # ------------------------------------------------------------------
    # 7. Feature importance (Random Forest) dan koefisien (Logistic Regression)
    # ------------------------------------------------------------------
    scaler = StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train)
    rf = RandomForestClassifier(random_state=42).fit(Xs, y_train)
    lr = LogisticRegression().fit(Xs, y_train)
    fi = pd.DataFrame({"Feature": FEATURES, "RF_Importance": rf.feature_importances_,
                       "LR_Coefficient": lr.coef_[0]}).sort_values("RF_Importance", ascending=False)
    print("\n7. FEATURE IMPORTANCE")
    print(fi.round(4).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    ax[0].barh(fi["Feature"][::-1], fi["RF_Importance"][::-1], color="#2b6cb0")
    ax[0].set_title("Feature Importance (Random Forest)")
    ax[0].set_xlabel("Importance Score")
    fc = fi.reindex(fi["LR_Coefficient"].abs().sort_values().index)
    ax[1].barh(fc["Feature"], fc["LR_Coefficient"],
               color=["#dd6b20" if v > 0 else "#2b6cb0" for v in fc["LR_Coefficient"]])
    ax[1].set_title("Koefisien (Logistic Regression)")
    ax[1].set_xlabel("Nilai Koefisien")
    save_fig("feature_importance.png")

    # ------------------------------------------------------------------
    # 9. Simpan model + info untuk dashboard
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("9. SIMPAN MODEL")
    print("=" * 60)
    model_path = os.path.join(OUT_DIR, "heart_model.joblib")
    joblib.dump(best_model, model_path)   # Pipeline: StandardScaler + classifier

    info = {
        "best_model": best_name,
        "features": FEATURES,
        "comparison": comparison.round(4).to_dict(orient="records"),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test, y_pred_best, target_names=["Survived", "Deceased"], output_dict=True, zero_division=0),
        "feature_importance": fi.round(4).to_dict(orient="records"),
        "n_total": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "class_counts": {"survived": int(class_counts.get(0, 0)), "deceased": int(class_counts.get(1, 0))},
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    with open(os.path.join(OUT_DIR, "model_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print("Model disimpan di:", model_path)

    # salin otomatis ke folder aplikasi Flask jika ada
    if os.path.isfile(os.path.join(WEB_DIR, "heart_api.py")):
        shutil.copy(model_path, WEB_DIR)
        shutil.copy(os.path.join(OUT_DIR, "model_info.json"), WEB_DIR)
        web_img = os.path.join(WEB_DIR, "static", "img")
        os.makedirs(web_img, exist_ok=True)
        for fn in os.listdir(IMG_DIR):
            shutil.copy(os.path.join(IMG_DIR, fn), web_img)
        print("File model, info, dan grafik sudah disalin ke:", WEB_DIR)
    else:
        print("Folder aplikasi Flask tidak ditemukan. Salin manual heart_model.joblib, "
              "model_info.json, dan folder img ke folder aplikasi Flask (heart_failure).")


if __name__ == "__main__":
    main()
