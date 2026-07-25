"""
Aplikasi Streamlit: Prediksi Segmen Pelanggan (Customer Segmentation).

Sebuah perusahaan otomotif ingin masuk ke pasar baru. Di pasar lama, pelanggan sudah
dikelompokkan ke 4 segmen (A, B, C, D) dan tiap segmen didekati dengan strategi
pemasaran berbeda. Aplikasi ini memprediksi segmen pelanggan baru menggunakan model
Random Forest, supaya strategi yang sama bisa diterapkan.

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
CATEGORICAL = ["Gender", "Ever_Married", "Graduated", "Profession", "Spending_Score"]
# Var_1 sengaja tidak dipakai: fitur anonim, dan terbukti sedikit menurunkan akurasi.
FEATURES = [
    "Gender", "Ever_Married", "Age", "Graduated",
    "Profession", "Work_Experience", "Spending_Score", "Family_Size",
]

# Spending_Score bersifat ordinal, diberi urutan bermakna (bukan alfabet).
SPENDING_ORDER = {"Low": 0, "Average": 1, "High": 2}

# Parameter terbaik hasil GridSearchCV di notebook.
RF_PARAMS = dict(
    n_estimators=100, max_depth=10, max_features="sqrt",
    min_samples_leaf=2, min_samples_split=2, criterion="gini", random_state=42,
)

# Label ramah untuk ditampilkan (fitur asli berbahasa Inggris).
LABELS = {
    "Gender": "Jenis Kelamin",
    "Ever_Married": "Pernah Menikah?",
    "Age": "Umur",
    "Graduated": "Lulus Kuliah?",
    "Profession": "Profesi",
    "Work_Experience": "Pengalaman Kerja (th)",
    "Spending_Score": "Skor Belanja",
    "Family_Size": "Jumlah Anggota Keluarga",
}


# ----------------------------------------------------------------------------
# Load data + latih model (di-cache supaya tidak diulang tiap interaksi)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_and_train():
    raw = pd.read_csv("train.csv").drop(columns=["ID"])

    # Nilai pengisi missing value (dihitung sekali, dipakai ulang untuk data baru).
    medians = {c: raw[c].median() for c in NUMERIC}
    modes = {c: raw[c].mode()[0] for c in CATEGORICAL}

    dfi = raw.copy()  # data terisi, kategori masih berupa teks (untuk profil)
    for c in NUMERIC:
        dfi[c] = dfi[c].fillna(medians[c])
    for c in CATEGORICAL:
        dfi[c] = dfi[c].fillna(modes[c])

    encoders = {}
    for c in CATEGORICAL:
        if c == "Spending_Score":
            encoders[c] = dict(SPENDING_ORDER)
        else:
            encoders[c] = {v: i for i, v in enumerate(sorted(dfi[c].unique()))}

    X = dfi[FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = X[c].map(encoders[c])
    y = dfi[TARGET]

    # Akurasi holdout (jujur, untuk ditampilkan).
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    acc = accuracy_score(y_te, RandomForestClassifier(**RF_PARAMS).fit(X_tr, y_tr).predict(X_te))

    model = RandomForestClassifier(**RF_PARAMS).fit(X, y)  # model final: seluruh data
    options = {c: list(encoders[c].keys()) for c in CATEGORICAL}
    return model, encoders, options, medians, modes, dfi, acc


model, encoders, options, medians, modes, dfi, acc = load_and_train()


def encode_frame(frame):
    """Isi missing value + encode kategorikal memakai statistik data latih."""
    out = frame.copy()
    for c in NUMERIC:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(medians[c])
    for c in CATEGORICAL:
        out[c] = out[c].fillna(modes[c]).map(encoders[c])
    return out[FEATURES].fillna(0)  # kategori tak dikenal -> 0


# Ciri khas tiap segmen, diturunkan dari data (bukan hardcode) — untuk penjelasan non-teknis.
avg_age, avg_fam = dfi["Age"].mean(), dfi["Family_Size"].mean()


def segment_traits(seg):
    row = dfi[dfi[TARGET] == seg]
    age, fam = row["Age"].mean(), row["Family_Size"].mean()
    spend = row["Spending_Score"].mode()[0]
    if age >= avg_age + 2:
        usia = f"cenderung lebih tua (rata-rata {age:.0f} th)"
    elif age <= avg_age - 2:
        usia = f"cenderung lebih muda (rata-rata {age:.0f} th)"
    else:
        usia = f"usia menengah (rata-rata {age:.0f} th)"
    keluarga = "keluarga cenderung besar" if fam >= avg_fam + 0.3 else (
        "keluarga cenderung kecil" if fam <= avg_fam - 0.3 else "ukuran keluarga sedang")
    return f"{usia}, {keluarga}, skor belanja umumnya **{spend}**."


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🛍️ Prediksi Segmen Pelanggan")
st.write(
    "Setiap pelanggan itu beda-beda. Perusahaan mengelompokkan pelanggan ke **4 segmen "
    "(A, B, C, D)** supaya bisa memberi strategi pemasaran yang pas untuk tiap kelompok. "
    "Aplikasi ini menebak segmen seorang pelanggan dari data dirinya."
)
st.caption(
    f"Model: Random Forest • Akurasi **{acc:.0%}** — diukur dari data latih berlabel "
    "(patokan kualitas model, bukan akurasi data yang kamu upload)."
)


# ----------------------------------------------------------------------------
# Sidebar: input satu pelanggan
# ----------------------------------------------------------------------------
st.sidebar.header("👤 Isi Data Pelanggan")
user = {
    "Gender": st.sidebar.selectbox(LABELS["Gender"], options["Gender"]),
    "Ever_Married": st.sidebar.selectbox(LABELS["Ever_Married"], options["Ever_Married"]),
    "Age": st.sidebar.slider(LABELS["Age"], 18, 90, 40),
    "Graduated": st.sidebar.selectbox(LABELS["Graduated"], options["Graduated"]),
    "Profession": st.sidebar.selectbox(LABELS["Profession"], options["Profession"]),
    "Work_Experience": st.sidebar.slider(LABELS["Work_Experience"], 0, 14, 1),
    "Spending_Score": st.sidebar.selectbox(LABELS["Spending_Score"], options["Spending_Score"]),
    "Family_Size": st.sidebar.slider(LABELS["Family_Size"], 1, 9, 3),
}

# ----------------------------------------------------------------------------
# Prediksi satu pelanggan
# ----------------------------------------------------------------------------
X_user = encode_frame(pd.DataFrame([user]))
pred = model.predict(X_user)[0]
proba = model.predict_proba(X_user)[0]
confidence = proba.max()

st.subheader("🎯 Hasil Prediksi")
st.success(f"Pelanggan ini paling cocok masuk **Segmen {pred}**  ·  keyakinan model {confidence:.0%}")
st.write(f"**Ciri khas Segmen {pred}:** {segment_traits(pred)}")

with st.expander("Lihat rincian data yang kamu isi & probabilitas tiap segmen"):
    nice = {LABELS[k]: v for k, v in user.items()}
    st.dataframe(pd.DataFrame([nice]), use_container_width=True)
    st.write("Seberapa yakin model ke tiap segmen:")
    st.bar_chart(pd.Series(proba, index=model.classes_, name="Probabilitas"))

# ----------------------------------------------------------------------------
# Profil tiap segmen (interpretasi untuk tim marketing)
# ----------------------------------------------------------------------------
st.subheader("📊 Karakter Tiap Segmen")
st.write("Ini rata-rata tiap segmen di data — membantu tim marketing paham 'segmen ini tipe orang seperti apa'.")
profile = dfi.groupby(TARGET).agg(
    Umur=("Age", "mean"),
    Pengalaman_Kerja=("Work_Experience", "mean"),
    Ukuran_Keluarga=("Family_Size", "mean"),
).round(1)
profile["Skor_Belanja_Umum"] = dfi.groupby(TARGET)["Spending_Score"].agg(lambda s: s.mode()[0])
st.dataframe(
    profile.style.apply(
        lambda r: ["background-color: #1b5e20" if r.name == pred else "" for _ in r], axis=1
    ),
    use_container_width=True,
)
st.caption(f"Baris tersorot = segmen hasil prediksi pelanggan di atas (Segmen {pred}).")

# ----------------------------------------------------------------------------
# Prediksi banyak pelanggan sekaligus (data contoh ATAU upload sendiri)
# ----------------------------------------------------------------------------
st.subheader("📦 Prediksi Banyak Pelanggan Sekaligus")
st.write(
    "Punya daftar pelanggan sendiri? Upload file CSV-nya untuk memprediksi semuanya "
    "sekaligus — atau coba dulu dengan data contoh."
)
st.info(
    f"ℹ️ **Soal akurasi:** angka {acc:.0%} di atas diukur dari data latih yang punya jawaban benar. "
    "Untuk pelanggan baru tanpa label, hasil di bawah adalah **tebakan terbaik** model — "
    "diperkirakan mirip selama karakter pelanggannya tidak jauh beda dari data latih."
)

mode = st.radio(
    "Sumber data:",
    ["🎲 Pakai data contoh (2.627 pelanggan baru)", "📤 Upload file CSV saya sendiri"],
)

source = None
if mode.startswith("🎲"):
    source = pd.read_csv("test.csv")
else:
    st.caption("File CSV harus punya kolom: " + ", ".join(FEATURES) + " (kolom ID opsional).")
    uploaded = st.file_uploader("Pilih file CSV", type=["csv"])
    if uploaded is not None:
        try:
            source = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")

if source is not None:
    missing = [c for c in FEATURES if c not in source.columns]
    if missing:
        st.error("File-nya kurang kolom berikut: " + ", ".join(missing))
    else:
        ids = source["ID"] if "ID" in source.columns else range(1, len(source) + 1)
        result = pd.DataFrame({"ID": ids, "Prediksi_Segmen": model.predict(encode_frame(source))})
        st.write(f"✅ Berhasil memprediksi **{len(result)} pelanggan**. Sebaran segmennya:")
        st.bar_chart(result["Prediksi_Segmen"].value_counts().sort_index())
        st.dataframe(result.head(20), use_container_width=True)
        st.download_button(
            "⬇️ Download hasil lengkap (CSV)",
            result.to_csv(index=False).encode("utf-8"),
            file_name="hasil_prediksi_segmen.csv",
            mime="text/csv",
        )

st.caption(
    "Dibuat dengan Streamlit • Dataset: Kaggle Customer Segmentation • "
    "Model: Random Forest (8 fitur, hyperparameter hasil GridSearchCV)."
)
