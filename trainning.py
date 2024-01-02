import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

def prepare(ds):
    batch_size = 32
    AUTOTUNE = tf.data.AUTOTUNE
    
    ds = tf.data.Dataset.from_tensor_slices((ds['Pixels'].values, ds['Labels'].values))

    data_augmentation = tf.keras.Sequential([
      layers.Reshape((48, 48, 3), input_shape=(48 * 48 * 3,)),
      layers.RandomContrast(factor=1),
      layers.RandomBrightness(factor=0.2),
    ])
  
    ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                num_parallel_calls=AUTOTUNE)
    
    # Use buffered prefetching
    prepared_ds = ds.prefetch(buffer_size=AUTOTUNE)
    augmented_df = pd.DataFrame(list(prepared_ds.as_numpy_iterator()), columns=['Pixels', 'Labels'])
    return augmented_df

#dataset ops - train_ds, val_ds, test_ds
dataset_path = 'P:/RGBdirections.csv'
image_size = (48,48,3)

def load(dataset_path, image_size):
    data = pd.read_csv(dataset_path)
    data = data.dropna()
    pixels = data['Pixels'].tolist()
    labels = data['Labels'].tolist()
    pix_images = []

    for pixel_sequence in pixels:
        # Check if the pixel_sequence is a float (skip it if it is)
        if isinstance(pixel_sequence, float):
            continue

        # Split pixel values and convert to integers
        pixel_values = [int(pixel) for pixel in pixel_sequence.split(' ')]
        # Reshape the pixel values into a 3D array
        pix_image = np.asarray(pixel_values).reshape(*image_size)
        pix_images.append(pix_image.astype('float32'))

    pix_images = np.asarray(pix_images)
    return pix_images, labels

pix_images, labels = load(dataset_path, image_size)

# Encode labels using LabelEncoder
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Convert encoded labels to one-hot encoding
one_hot_labels = to_categorical(encoded_labels)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(pix_images, one_hot_labels,test_size=2000)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))  # 3 classes: right, left, straight

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
y_pred = model.predict(x_test)

loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

model.save('gesture_model.h5')
