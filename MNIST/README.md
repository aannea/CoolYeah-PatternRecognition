# MNIST: Klasifikasi Gambar Digit-digit Tulisan Tangan

## Deskripsi Proyek
Proyek ini adalah implementasi model deep learning untuk mengklasifikasikan gambar digit-digit tulisan tangan. Dataset yang digunakan adalah **MNIST** yang berisi 60,000 gambar untuk data pelatihan dan 10,000 gambar untuk data pengujian, masing-masing berupa gambar grayscale yang sudah berukuran 28x28 piksel.

## Fitur
- Pra-pemrosesan gambar
- Pelatihan model klasifikasi menggunakan Convolutional Neural Network (CNN)
- Evaluasi model
- Testing model

## Deksripsi Fitur
### 1. Preprocessing gambar
- Preprocessing gambar menggunakan transformasi dengan melakukan random terhadap rotasi gambar dalam rentang gambar -10 dan 10, kemudian dilakukan
  menerapkan transformasi dengan random affine dengan parameter pertama 0 dan translate 10% yang
  memungkinkan untuk diterjemahkan secara horizontal dan vertikal. Selanjutnya gambar ditransformasi ke bentuk tensor dan terakhir
  dilakukan normalisasi sebesar 0.5 antara mean dan std
### 2. Pelatihan model klasifikasi CNN
- model menggunakan konvolusi, pooling dan laposan fully connected (FC). dengan jaringan conv 2 lapisan
  menggunakan 32 dan 64 filter dengan ukuran kernel 3x3, diikuti pooling dengan max pooling 2x2, kemudian
  terdapat dua lapisan fully connected yakni fc1 yang menghubungkan 64x7x7 ke 128 dan fc2 128 ke 10 sesuai dengan jumlah kelas
  pada dataset, dropout probabilitas 0.25 untuk mengurangi overfitting.
- pada fungsi forward, conv pertama diikuti ReLU dan pooling, kemudian conv 2 dengan ReLU dan pooling.
  setelah itu, data di flatten untuk menjadi input bagi fc.
- fungsi loss menggunakan cross entropy loss
- optimizer menggunakan adam
### 3. Evaluasi model
- evaluasi pada dataset mnist 10K gambar test
- menghasilkan keluaran dengan bentuk akurasi
### 4. testing model
- gambar dipreprocessing dahulu dengan melakukan grayscalling, rezise gambar 28x28, dibuat dalam bentuk
  tensor, dan di normalize dengan 0.5 antara mean dan std
- gambar di load dan proses untuk melakukan testing prediksi terhadap gambar testing

## Instalasi

1. **Clone repository ini:**
   ```bash
   git clone https://github.com/aannea/CoolYeah-PatternRecognition.git
   cd MNIST
   ```
   
2. **Buat venv**
   ```bash
   python -m venv .env
   source env/bin/activate # untuk pengguna Unix
   .\env\Scripts\activate # untuk pengguna Windows
   ```
   
3. Install Dependensi

4. Pelatihan Model
```bash
   python train_mnist.py
   ```
5. Testing Model
```bash
   python predict_mnist.py
   ```


## Kontributor
1. Farhan Aryo Pangestu (train dan eval)
2. Annisa Febriana (cari dataset MNIST)
