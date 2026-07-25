"""
Aplikasi Streamlit: Prediksi Segmen Pelanggan (Customer Segmentation).

Sebuah perusahaan otomotif ingin memasukkan produknya ke pasar baru. Di pasar lama,
pelanggan sudah dikelompokkan ke 4 segmen (A, B, C, D) dan tiap segmen didekati dengan
strategi pemasaran berbeda. Aplikasi ini memprediksi segmen untuk pelanggan baru
menggunakan model Random Forest, supaya strategi yang sama bisa diterapkan.

Dataset: Kaggle "Customer Segmentation" (train.csv berlabel, test.csv = pelanggan baru).
"""

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ----------------------------------------------------------------------------
# Konfigurasi & konstanta
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Prediksi Segmen Pelanggan", page_icon="🛍️", layout="wide")

TARGET = "Segmentation"
NUMERIC = ["Age", "Work_Experience", "Family_Size"]
CATEGORICAL = ["Gender", "Ever_Married", "Graduated", "Profession", "Spending_Score", "Var_1"]
FEATURES = [
    "Gender", "Ever_Married", "Age", "Graduated", "Profession",
    "Work_Experience", "Spending_Score", "Family_Size", "Var_1",
]

# Spending_Score bersifat ordinal, jadi diberi urutan yang bermakna (bukan alfabet).
SPENDING_ORDER = {"Low": 0, "Average": 1, "High": 2}

# Parameter terbaik hasil GridSearchCV di notebook.
RF_PARAMS = dict(
    n_estimators=100, max_depth=10, max_features="sqrt",
    min_samples_leaf=2, min_samples_split=2, criterion="gini", random_state=42,
)


# ----------------------------------------------------------------------------
# Load data + latih model (di-cache supaya tidak diulang tiap interaksi)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_and_train():
    raw = pd.read_csv("train.csv").drop(columns=["ID"])

    # Nilai untuk mengisi missing value (dihitung dari data mentah, dipakai ulang
    # saat prediksi data baru agar konsisten).
    medians = {c: raw[c].median() for c in NUMERIC}
    modes = {c: raw[c].mode()[0] for c in CATEGORICAL}

    df = raw.copy()
    for c in NUMERIC:
        df[c] = df[c].fillna(medians[c])
    for c in CATEGORICAL:
        df[c] = df[c].fillna(modes[c])

    # Bangun encoder kategorikal -> angka (dipakai untuk data latih & input user).
    encoders = {}
    for c in CATEGORICAL:
        if c == "Spending_Score":
            encoders[c] = dict(SPENDING_ORDER)
        else:
            encoders[c] = {v: i for i, v in enumerate(sorted(df[c].unique()))}

    X = df[FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = X[c].map(encoders[c])
    y = df[TARGET]

    # Akurasi holdout (jujur, untuk ditampilkan ke user).
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    acc = accuracy_score(y_te, RandomForestClassifier(**RF_PARAMS).fit(X_tr, y_tr).predict(X_te))

    # Model final dilatih di SELURUH data latih.
    model = RandomForestClassifier(**RF_PARAMS).fit(X, y)

    options = {c: list(encoders[c].keys()) for c in CATEGORICAL}
    return model, encoders, options, medians, modes, df, acc


model, encoders, options, medians, modes, df, acc = load_and_train()


def encode_frame(frame):
    """Isi missing value + encode kategorikal memakai statistik data latih."""
    out = frame.copy()
    for c in NUMERIC:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(medians[c])
    for c in CATEGORICAL:
        out[c] = out[c].fillna(modes[c]).map(encoders[c])
    out = out[FEATURES]
    # Kategori yang tak dikenal -> NaN; isi 0 supaya model tetap bisa memprediksi.
    return out.fillna(0)


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🛍️ Prediksi Segmen Pelanggan")
st.write(
    "Memprediksi segmen pelanggan (**A / B / C / D**) dari data demografi & perilaku "
    "menggunakan **Random Forest**. Isi data pelanggan di sidebar kiri untuk memprediksi "
    "satu pelanggan, atau lakukan prediksi massal untuk seluruh pelanggan baru di bawah."
)
st.metric("Akurasi model (data uji holdout)", f"{acc:.1%}")


# ----------------------------------------------------------------------------
# Sidebar: input satu pelanggan
# ----------------------------------------------------------------------------
st.sidebar.header("👤 Data Pelanggan")
user = {
    "Gender": st.sidebar.selectbox("Gender", options["Gender"]),
    "Ever_Married": st.sidebar.selectbox("Ever Married (sudah menikah?)", options["Ever_Married"]),
    "Age": st.sidebar.slider("Age (umur)", 18, 90, 40),
    "Graduated": st.sidebar.selectbox("Graduated (lulus kuliah?)", options["Graduated"]),
    "Profession": st.sidebar.selectbox("Profession", options["Profession"]),
    "Work_Experience": st.sidebar.slider("Work Experience (tahun)", 0, 14, 1),
    "Spending_Score": st.sidebar.selectbox("Spending Score", options["Spending_Score"]),
    "Family_Size": st.sidebar.slider("Family Size", 1, 9, 3),
    "Var_1": st.sidebar.selectbox("Var_1 (kategori anonim)", options["Var_1"]),
}


# ----------------------------------------------------------------------------
# Prediksi satu pelanggan
# ----------------------------------------------------------------------------
X_user = encode_frame(pd.DataFrame([user]))
pred = model.predict(X_user)[0]
proba = model.predict_proba(X_user)[0]

st.subheader("📝 Data Input Kamu")
st.dataframe(pd.DataFrame([user]), use_container_width=True)

st.subheader("🎯 Hasil Prediksi")
st.success(f"Pelanggan ini diprediksi masuk ke **Segmen {pred}**")
st.write("Probabilitas tiap segmen:")
st.bar_chart(pd.Series(proba, index=model.classes_, name="Probabilitas"))


# ----------------------------------------------------------------------------
# Profil rata-rata tiap segmen (interpretasi)
# ----------------------------------------------------------------------------
st.subheader("📊 Profil Rata-Rata per Segmen")
profile = df.groupby(TARGET).agg(
    Umur=("Age", "mean"),
    Pengalaman_Kerja=("Work_Experience", "mean"),
    Ukuran_Keluarga=("Family_Size", "mean"),
).round(1)
st.dataframe(profile, use_container_width=True)


# ----------------------------------------------------------------------------
# Prediksi massal untuk pelanggan baru (test.csv)
# ----------------------------------------------------------------------------
st.subheader("📦 Prediksi Massal — Pelanggan Baru")
st.write(
    "Perusahaan punya **2.627 calon pelanggan baru** (tanpa label). Klik tombol di bawah "
    "untuk memprediksi segmen mereka semua sekaligus."
)
if st.button("🔮 Prediksi semua pelanggan baru (test.csv)"):
    new_customers = pd.read_csv("test.csv")
    ids = new_customers["ID"]
    X_new = encode_frame(new_customers.drop(columns=["ID"]))
    result = pd.DataFrame({"ID": ids, "Predicted_Segment": model.predict(X_new)})

    st.write(f"Berhasil memprediksi **{len(result)} pelanggan**. Distribusinya:")
    st.bar_chart(result["Predicted_Segment"].value_counts().sort_index())
    st.dataframe(result.head(20), use_container_width=True)
    st.download_button(
        "⬇️ Download hasil lengkap (CSV)",
        result.to_csv(index=False).encode("utf-8"),
        file_name="predicted_segments.csv",
        mime="text/csv",
    )

st.caption(
    "Dibuat dengan Streamlit • Dataset: Kaggle Customer Segmentation • "
    "Model: Random Forest (9 fitur, hyperparameter hasil GridSearchCV)."
)
