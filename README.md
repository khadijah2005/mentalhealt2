# 🧠 Mental Health Gen Z Monitoring Dashboard

## Deskripsi Proyek

Proyek ini dibuat untuk memonitor isu kesehatan mental pada Generasi Z berdasarkan data media sosial Twitter/X (menfess). Banyak anak muda meluapkan perasaan seperti kecemasan, depresi, burnout, atau kebutuhan untuk bercerita melalui media sosial. Sistem ini bertujuan mendeteksi tren isu kesehatan mental untuk meningkatkan kesadaran sosial.

Proyek ini dikembangkan menggunakan Python dan Streamlit untuk menghasilkan dashboard interaktif yang dapat membantu memvisualisasikan kondisi dan pola isu kesehatan mental pada Generasi Z.

---

## Tema Project

**TEMA 7: Monitoring Isu Kesehatan Mental (Mental Health) pada Generasi Z**

---

## Skenario Masalah

Banyak anak muda meluapkan kecemasan, depresi, atau burnout di media sosial. Proyek ini bertujuan mendeteksi tren isu kesehatan mental demi meningkatkan kesadaran sosial.

---

## Sumber Data

Data diperoleh dari Twitter/X Menfess (contoh: akun seperti @schfess dan akun sejenis).

Kata kunci yang digunakan:

* Burnout kuliah
* Depresi
* Anxiety
* Butuh teman cerita
* Mental Health

Dataset disimpan dalam format CSV untuk proses analisis lebih lanjut.

---

## Fitur Dashboard

### 1. Lampu Indikator Urgensi Isu

Mengidentifikasi peningkatan tingkat urgensi curhatan.

Kategori:

* Butuh Pertolongan Segera
* Curhat Ringan

Status indikator:

* 🟢 Aman
* 🟡 Perlu perhatian
* 🔴 Darurat

---

### 2. Klasifikasi Tingkat Urgensi Curhatan

Melakukan klasifikasi postingan menjadi:

* Butuh Pertolongan Segera
* Curhat Ringan

Visualisasi:

* Bar Chart
* Classification Report
* Confusion Matrix

---

### 3. Klasterisasi & Trending Topic

Mengelompokkan penyebab kecemasan menjadi beberapa klaster:

* Tekanan akademik
* Masalah keluarga
* Finansial

Visualisasi:

* Grafik lingkaran (Pie Chart)
* Word Cloud
* Trending Topic

---

### 4. Support System Network (SNA)

Menganalisis jaringan akun pendukung yang memberikan respons positif.

Analisis:

* Degree Centrality
* Betweenness Centrality
* Community Detection

Visualisasi:

* Network Graph
* Top akun pendukung

---

## Tools dan Library

Tools:

* Python
* Streamlit
* GitHub

Library:

* pandas
* numpy
* scikit-learn
* plotly
* networkx
* python-louvain
* matplotlib
* seaborn
* nltk
* Sastrawi
* wordcloud

---

## Struktur Folder Project

```text
mentalhealth/
│
├── app.py
├── requirements.txt
├── data.csv
├── assets/
│   └── logo.png
└── README.md
```

## Cara Menjalankan Project

Install library:

```bash
pip install -r requirements.txt
```

Jalankan Streamlit:

```bash
streamlit run app.py
```

---

## Pembagian Tugas Kelompok

1. Data Collection & Preprocessing
2. Modeling & Analisis
3. Dashboard & Presentasi

---

## Tujuan Project

Membantu meningkatkan kesadaran sosial terhadap isu kesehatan mental Generasi Z melalui visualisasi data dan analisis media sosial.

---

FTI — Tema 7 Monitoring Mental Health Gen Z
