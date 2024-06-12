# Deteksi Badai pada Permukaan Bumi Melalui Satelit

![Project Logo](https://github.com/aannea/CoolYeah-PatternRecognition/blob/main/deteksi_badai_melalui_satelit/kenanih.jpeg)


## Deskripsi Proyek
Proyek ini bertujuan untuk mendeteksi badai pada permukaan bumi menggunakan dataset 
gambar satelit dengan menerapkan model Convolutional Neural Network (CNN). Proyek ini 
melibatkan berbagai tahapan termasuk pra-pemrosesan gambar, pelatihan model, evaluasi, dan 
pengujian model.


## Fitur
- Pra-pemrosesan gambar
- Pelatihan model klasifikasi menggunakan Convolutional Neural Network (CNN)
- Evaluasi model
- Pengujian model

## Deskripsi Fitur

### 1. Pra-pemrosesan Gambar
- Normalisasi gambar dengan membagi nilai piksel dengan 255 untuk mendapatkan nilai antara 0 dan 1.
- Mengubah gambar ke ukuran 128x128 piksel.
- Membuat batch gambar untuk memudahkan proses pelatihan dan evaluasi.

### 2. Pelatihan Model Klasifikasi CNN
- Menggunakan beberapa lapisan konvolusi dengan aktivasi ReLU dan pooling untuk mengekstraksi fitur dari gambar.
- Menggunakan lapisan fully connected dan dropout untuk mencegah overfitting.
- Melatih model dengan optimizer Adam dan loss function binary crossentropy.

### 3. Evaluasi Model
- Mengevaluasi kinerja model dengan menggunakan dataset validasi.
- Mengukur akurasi model selama pelatihan.

### 4. Pengujian Model
- Menggunakan model terlatih untuk melakukan prediksi pada gambar baru.
- Menyimpan model terlatih untuk digunakan kembali di masa mendatang.

## Instalasi

1. **Clone repository ini:**
   ```bash
   git clone https://github.com/aannea/CoolYeah-PatternRecognition.git
   cd deteksi_badai_melalui_satelit
   ```
   
2. **Buat venv**
   ```bash
   python -m venv .env
   source env/bin/activate # untuk pengguna Unix
   .\env\Scripts\activate # untuk pengguna Windows
   ```
   
3. Install Dependensi

4. Pelatihan Model
   ```python
   python train.py
   ```

5. Testing Model
   ```python
   python test.py
   ```


## Kontributor
1. Amalia Suciati (cari data badai, kontribusi train badai)
2. Bintang Rizqi Pasha (kontri predict badai,)
