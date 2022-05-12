import os

import numpy as np
from keras.engine.input_layer import InputLayer
from keras.layers import Conv2D, UpSampling2D
from keras.preprocessing.image import img_to_array, load_img
from skimage.transform import resize
from skimage.color import rgb2lab, lab2rgb
from keras.models import Sequential
from skimage.io import imsave


def generate_train_data():
    images = []
    path = 'resources/train/'
    for filename in os.listdir(path):
        img = img_to_array(load_img(path + filename))
        img = resize(img, (256, 256, 3))
        images.append(img)
    x = []
    y = []
    for image in images:
        image = np.array(image, dtype=float)
        x.append((rgb2lab(1.0 / 255 * image)[:, :, 0]).reshape(256, 256, 1))
        y.append((rgb2lab(1.0 / 255 * image)[:, :, 1:]) / 128)

    x = np.asarray(x)
    y = np.asarray(y)

    return x, y


def build_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(256, 256, 1)))

    model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))

    model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

    model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))

    model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))

    model.compile(optimizer='rmsprop', loss='mse')
    return model


def load_test_data():
    data = []
    path = 'resources/test/'
    for filename in os.listdir(path):
        img = img_to_array(load_img(path + filename))
        img = resize(img, (256, 256, 3))
        img = np.array(img, dtype=float)
        data.append((rgb2lab(1.0 / 255 * img)[:, :, 0]).reshape(256, 256, 1))

    data = np.asarray(data)
    return data


def generate_images(initial_data, predicted_data):
    image = np.zeros((256, 256, 3))
    i = 0
    for x, y in zip(initial_data, predicted_data):
        image[:, :, 0] = x[:, :, 0]
        image[:, :, 1:] = y
        imsave(f'resources/result/out_{i}.png', lab2rgb(image))
        i += 1


def predict(model, x_train):
    return model.predict(x_train) * 128


def run():
    model = build_model()
    x, y = generate_train_data()

    model.fit(x=x, y=y, batch_size=10, epochs=100)
    print(model.evaluate(x, y, batch_size=10))
    x_train = load_test_data()

    output = predict(model, x_train)

    generate_images(x_train, output)


run()
