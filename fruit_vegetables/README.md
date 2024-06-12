# Deteksi Buah dan Sayuran Menggunakan Vision Transformer (ViT)

<img src="https://github.com/aannea/CoolYeah-PatternRecognition/blob/main/fruit_vegetables/img.png"
  height="200">

## Deskripsi Proyek
Proyek ini bertujuan untuk mengklasifikasikan gambar buah dan sayuran menggunakan model Vision Transformer (ViT) yang telah dilatih sebelumnya. Proyek ini melibatkan beberapa tahap penting, termasuk pra-pemrosesan gambar, pelatihan model, evaluasi, dan pengujian model.

## Fitur
- Pra-pemrosesan gambar
- Pelatihan model klasifikasi menggunakan Vision Transformer (ViT)
- Evaluasi model
- Pengujian model

## Deskripsi Fitur

### 1. Preprocessing Gambar
- **RandomResizedCrop:** Mengubah ukuran gambar secara acak menjadi 224x224 piksel.
- **RandomHorizontalFlip:** Melakukan flipping horizontal pada gambar secara acak.
- **ColorJitter:** Menyesuaikan kecerahan, kontras, saturasi, dan hue gambar secara acak.
- **ToTensor:** Mengonversi gambar menjadi bentuk tensor.
- **Normalize:** Normalisasi nilai piksel gambar dengan mean dan standar deviasi.

### 2. Pelatihan Model Klasifikasi ViT
- Menggunakan model Vision Transformer (ViT) yang telah dilatih sebelumnya (pretrained).
- Mengubah lapisan terakhir (head) dari model untuk disesuaikan dengan jumlah kelas pada dataset.
- Melatih model menggunakan optimizer Adam dan fungsi loss CrossEntropy.
- Menyimpan model terlatih untuk digunakan kembali di masa mendatang.

### 3. Evaluasi Model
- Mengukur kinerja model dengan menggunakan dataset validasi.
- Menghitung nilai loss dan akurasi selama pelatihan dan validasi.

### 4. Pengujian Model
- Menggunakan model terlatih untuk melakukan prediksi pada gambar baru.

## Instalasi

1. **Clone repository ini:**
   ```bash
   git clone https://github.com/aannea/CoolYeah-PatternRecognition.git
   cd fruit_vegetables
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
   python predic.py
   ```

## Kontributor
1. Anisa Febriana (Kontri predict.py buah sayur)
2. Bintang Rizqi Pasha (cari dataset fruit_vegetables, kontri train.py buah sayur)

## Citation
```
Kritik Seth, "Fruits and Vegetables Image Recognition Dataset," Kaggle 2020 [https://www.kaggle.com/kritikseth/fruit-and-vegetable-image-recognition]
```
