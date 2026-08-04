# 🚚 Optimasi VRPTW: Solomon I1 vs Dragonfly Algorithm vs Hybrid NN-DA

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Skripsi%20%2F%20Final%20Project-yellow)

Proyek skripsi yang membandingkan performa tiga pendekatan penyelesaian **Vehicle Routing Problem with Time Windows (VRPTW)**:

1. **Solomon I1 Insertion Heuristic** — heuristik klasik berbasis penyisipan (baseline).
2. **Dragonfly Algorithm (DA)** — algoritma metaheuristik terinspirasi perilaku kawanan capung, dijalankan murni dari populasi acak.
3. **Hybrid Nearest Neighbor + Dragonfly Algorithm (NN + DA)** — solusi awal dari Nearest Neighbor construction digunakan sebagai *seed* populasi awal DA.

Aplikasi dibangun dengan **Streamlit** untuk mempermudah pengujian, visualisasi rute, dan perbandingan hasil secara interaktif.

<!--
📸 TODO: Tambahkan screenshot atau GIF demo aplikasi di sini.
Contoh:
![Demo Aplikasi](assets/demo.gif)
-->

---

## 📋 Daftar Isi

- [Latar Belakang](#-latar-belakang)
- [Metode yang Dibandingkan](#-metode-yang-dibandingkan)
- [Struktur Proyek](#-struktur-proyek)
- [Instalasi](#-instalasi)
- [Cara Menjalankan](#-cara-menjalankan)
- [Format Dataset](#-format-dataset)
- [Metrik Evaluasi](#-metrik-evaluasi)
- [Referensi](#-referensi)
- [Lisensi](#-lisensi)

---

## 🧩 Latar Belakang

**VRPTW (Vehicle Routing Problem with Time Windows)** adalah perluasan dari Vehicle Routing Problem klasik, di mana setiap pelanggan memiliki jendela waktu (*time window*) tertentu untuk dilayani, selain batasan kapasitas kendaraan. Masalah ini termasuk kategori NP-hard, sehingga pendekatan heuristik dan metaheuristik banyak digunakan untuk mencari solusi mendekati optimal dalam waktu yang wajar.

Proyek ini menguji apakah menggabungkan solusi awal dari **Nearest Neighbor Construction** sebagai *seed* populasi **Dragonfly Algorithm** dapat menghasilkan performa yang lebih baik dibandingkan menjalankan Dragonfly Algorithm secara murni (populasi acak) maupun dibandingkan baseline Solomon I1 Insertion.

## 🔬 Metode yang Dibandingkan

| Metode | Deskripsi |
|---|---|
| **Solomon I1 Insertion** | Heuristik konstruktif klasik, deterministik, dijalankan satu kali sebagai baseline. |
| **Dragonfly Algorithm (murni)** | Metaheuristik berbasis populasi terinspirasi perilaku kawanan capung (separation, alignment, cohesion, food factor, enemy factor), populasi awal acak. |
| **Hybrid NN + Dragonfly Algorithm** | Populasi awal DA diturunkan dari solusi Nearest Neighbor Construction (encoded sebagai *seed vector*). |

Setiap metode dievaluasi berdasarkan: jumlah kendaraan, total jarak tempuh, dan total waktu kunjungan, lalu dibandingkan dalam bentuk tabel dan gap (%) terhadap baseline Solomon I1.

## 📁 Struktur Proyek

```
├── vrptw_core.py          # Modul inti: struktur data, pembacaan dataset,
│                          #   Solomon I1, Nearest Neighbor, Dragonfly Algorithm,
│                          #   evaluasi fitness, dan visualisasi rute
├── app_unit_vrptw.py       # Antarmuka Streamlit untuk pengujian & perbandingan
├── dataset/                # Folder dataset VRPTW (format CSV, benchmark Solomon)
├── assets/                 # Screenshot/GIF demo aplikasi (untuk README)
├── requirements.txt        # Daftar dependensi Python
└── README.md
```

## ⚙️ Instalasi

```bash
# 1. Clone repository ini
git clone https://github.com/USERNAME/NAMA-REPO.git
cd NAMA-REPO

# 2. (Opsional) buat virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 3. Install dependensi
pip install -r requirements.txt
```

## ▶️ Cara Menjalankan

```bash
streamlit run app_unit_vrptw.py
```

Aplikasi akan terbuka otomatis di browser (`http://localhost:8501`). Langkah penggunaan:

1. **Tahap 1 — Input data**: pilih dataset, jumlah customer, dan parameter Dragonfly Algorithm (jumlah run, iterasi maksimum, jumlah agen).
2. Klik **▶ Mulai pengujian**.
3. **Tahap 3 — Hasil**: lihat tabel ringkasan, perbandingan gap performa, dan visualisasi rute masing-masing metode.

> 💡 Pastikan dataset (.csv) sudah ditempatkan di folder `dataset/` pada root repo agar terbaca otomatis.

## 📊 Format Dataset

Dataset menggunakan format CSV benchmark Solomon dengan kolom (nama kolom fleksibel/alias didukung):

| Kolom | Keterangan |
|---|---|
| `id` / `CustNo` | Nomor identitas customer (depot memiliki `demand = 0`) |
| `x`, `y` | Koordinat lokasi |
| `demand` | Permintaan/kapasitas yang dibutuhkan |
| `ready` | Waktu paling awal customer bisa dilayani |
| `due` | Batas waktu akhir customer harus dilayani |
| `service` | Durasi waktu pelayanan |

## 📈 Metrik Evaluasi

- **Jumlah kendaraan** yang digunakan
- **Total jarak tempuh** seluruh rute
- **Total waktu kunjungan**
- **Waktu komputasi** tiap metode
- **Gap (%)** performa DA & Hybrid NN+DA terhadap baseline Solomon I1

## 📚 Referensi

- Solomon, M. M. (1987). *Algorithms for the vehicle routing and scheduling problems with time window constraints.*
- Mirjalili, S. (2016). *Dragonfly algorithm: a new meta-heuristic optimization technique for solving single-objective, discrete, and multi-objective problems.*

## 📄 Lisensi

Proyek ini dirilis di bawah lisensi [MIT](LICENSE) — bebas digunakan untuk keperluan riset, pembelajaran, atau pengembangan lebih lanjut dengan tetap mencantumkan atribusi.

---

<p align="center">Dibuat sebagai bagian dari Tugas Akhir/Skripsi 🎓</p>
