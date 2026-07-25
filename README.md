# 🛍️ Customer Segmentation (Multi-Class Classification)

Aplikasi web interaktif yang memprediksi **segmen pelanggan (A / B / C / D)** dari data demografi & perilaku menggunakan **Random Forest**. Dibangun dari kasus nyata: sebuah perusahaan otomotif ingin masuk ke pasar baru dan menerapkan strategi pemasaran tersegmentasi ke ~2.600 calon pelanggan baru.

> 🔗 **Live demo:** _(isi setelah deploy)_ https://xxxxx.streamlit.app

<!-- Tips: tambahkan screenshot aplikasi di sini setelah deploy -->
<!-- ![Screenshot](screenshot.png) -->

---

## ✨ Fitur
- **Prediksi 1 pelanggan** — isi data lewat sidebar, dapat segmen + probabilitas tiap kelas.
- **Prediksi massal** — sekali klik memprediksi seluruh pelanggan baru (`test.csv`) + tombol download hasil CSV.
- **Profil segmen** — rata-rata umur, pengalaman kerja, & ukuran keluarga per segmen.
- **Akurasi model** ditampilkan langsung (dievaluasi pada data uji holdout).

## 📊 Dataset
[Kaggle — Customer Segmentation](https://www.kaggle.com/datasets/abisheksudarshan/customer-segmentation). `train.csv` (8.068 baris, berlabel) & `test.csv` (2.627 pelanggan baru tanpa label). 9 fitur:

| Fitur | Keterangan |
|---|---|
| `Gender` | Jenis kelamin |
| `Ever_Married` | Status pernah menikah |
| `Age` | Umur |
| `Graduated` | Status lulus kuliah |
| `Profession` | Profesi (9 kategori) |
| `Work_Experience` | Pengalaman kerja (tahun) |
| `Spending_Score` | Skor belanja (Low / Average / High) |
| `Family_Size` | Jumlah anggota keluarga |
| `Var_1` | Kategori anonim (Cat_1–Cat_7) |

Target: **`Segmentation`** (A, B, C, D).

## 🧠 Metode
1. **Preprocessing** — buang `ID`, isi missing value (median untuk numerik, modus untuk kategorikal).
2. **Encoding** — fitur kategorikal → numerik; `Spending_Score` diberi urutan ordinal (Low < Average < High).
3. **Model** — `RandomForestClassifier` dengan hyperparameter hasil `GridSearchCV`.
4. **Evaluasi** — akurasi diukur pada data uji holdout (train/test split berstrata).

> Model membandingkan Random Forest, SVM, KNN, dan Decision Tree pada tahap eksplorasi; Random Forest dipilih sebagai yang terbaik lalu di-tuning.

## 🛠️ Tech Stack
Python · Streamlit · scikit-learn · pandas · NumPy

## 🚀 Menjalankan secara lokal
```bash
git clone https://github.com/byochiram/customer-segmentation.git
cd customer-segmentation
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Struktur
```
customer-segmentation/
├── app.py              # Aplikasi Streamlit
├── train.csv           # Data latih (berlabel)
├── test.csv            # Pelanggan baru (untuk prediksi massal)
├── requirements.txt    # Dependensi
└── README.md
```
