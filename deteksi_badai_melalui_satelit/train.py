#mengimpor libary tensorflow
import tensorflow as tf

#fungsi preprocess
#enormalisasi gambar dengan membagi nilai pixelnya dengan 255.
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

#Mendefinisikan ukuran gambar
def solution_model():
    IMG_SIZE = 128
    BATCH_SIZE = 64

#Memuat dataset gambar untuk pelatihan (train_ds) dan validasi (val_ds) dari direktori yang ditentukan
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory='train/',
        image_size= (IMG_SIZE, IMG_SIZE)
        , batch_size=BATCH_SIZE, shuffle=True)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory='validation/',
        image_size=  (IMG_SIZE, IMG_SIZE)
        , batch_size=BATCH_SIZE)

#memproses dataset
#Memetakan fungsi preprocess ke dataset pelatihan dan validasi.
    train_ds = train_ds.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
        tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

#membangun model
    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

#Melatih model menggunakan dataset pelatihan dan validasi selama 10 epoch
    model.fit(

        train_ds,
        validation_data=val_ds,
        epochs=10
    )

#menyimpan model
    return model

if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")

