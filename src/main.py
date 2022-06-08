import os

import numpy as np
from keras.engine.input_layer import InputLayer
from keras.layers import Conv2D, UpSampling2D, LayerNormalization, Conv2DTranspose
from keras.preprocessing.image import img_to_array, load_img
from skimage.transform import resize
from skimage.color import rgb2lab, lab2rgb
from keras.models import Sequential
from skimage.io import imsave

from src.graphs import generate_graph

from keras.preprocessing.image import ImageDataGenerator

img_width = 128
img_height = 128


def generate_train_data():
    images = []
    path = '../resources/train/'
    print('generating train data...')
    for filename in os.listdir(path):
        img = img_to_array(load_img(path + filename))
        print(path + filename)
        img = resize(img, (img_width, img_height, 3))
        images.append(img)
    x = []
    y = []
    for image in images:
        image = np.array(image, dtype=float)
        x.append((rgb2lab(1.0 / 255 * image)[:, :, 0]).reshape(img_width, img_height, 1))
        y.append((rgb2lab(1.0 / 255 * image)[:, :, 1:]) / 128)

    x = np.asarray(x)
    y = np.asarray(y)

    return x, y


def build_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(img_width, img_height, 1)))

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

    model.compile(optimizer='rmsprop', loss='mse', metrics=['acc'])
    model.summary()
    return model


def build_model_eccv():
    model = Sequential()

    model.add(InputLayer(input_shape=(img_width, img_height, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=2, padding='same'))
    model.add(LayerNormalization())

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', strides=2, padding='same'))
    model.add(LayerNormalization())

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
    model.add(LayerNormalization())

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(LayerNormalization())

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=2))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=2))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=2))
    model.add(LayerNormalization())

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=2))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=2))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=2))
    model.add(LayerNormalization())

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=2))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=2))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=2))
    model.add(LayerNormalization())

    model.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(313, (1, 1), padding='same'))
    model.add(Conv2D(2, (1, 1), padding='same', dilation_rate=1))
    model.add(UpSampling2D((4, 4), interpolation='bilinear'))

    model.summary()
    model.compile(optimizer='rmsprop', loss='mse', metrics=['acc'])
    return model


def build_model_3():
    model = Sequential()
    model.add(InputLayer(input_shape=(img_width, img_height, 1)))

    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

    # model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

    # model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))

    # model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))

    model.compile(optimizer='rmsprop', loss='mse')
    return model


def load_test_data():
    data = []
    path = '../resources/test/'
    for filename in os.listdir(path):
        img = img_to_array(load_img(path + filename))
        img = resize(img, (img_width, img_height, 3))
        img = np.array(img, dtype=float)
        data.append((rgb2lab(1.0 / 255 * img)[:, :, 0]).reshape(img_width, img_height, 1))

    data = np.asarray(data)
    return data


def generate_images(initial_data, predicted_data):
    image = np.zeros((img_width, img_height, 3))
    i = 0
    for x, y in zip(initial_data, predicted_data):
        image[:, :, 0] = x[:, :, 0]
        image[:, :, 1:] = y
        imsave(f'../resources/result/out_{i}.png', lab2rgb(image))
        i += 1


def predict(model, x_train):
    return model.predict(x_train) * 128


transformedImgGenerator = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    horizontal_flip=True
)


def imageGenerator(X, Y):
    for data in transformedImgGenerator.flow(X, Y, batch_size=20):
        yield data


def run(batch_size, epochs):
    model = build_model_eccv()
    x, y = generate_train_data()

    slicePos = int(0.8 * len(x))
    x_train = x[:slicePos]
    y_train = y[:slicePos]

    x_validate = x[slicePos:]
    y_validate = y[slicePos:]

    # history = model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    history = model.fit(imageGenerator(x_train, y_train), validation_data=imageGenerator(x_validate, y_validate),
                        steps_per_epoch=84, validation_steps=21, epochs=epochs)
    print(model.evaluate(x, y, batch_size=batch_size))
    x_train = load_test_data()

    output = predict(model, x_train)

    generate_images(x_train, output)
    generate_graph(history.history['acc'], history.history['val_acc'], history.history['loss'],
                   history.history['val_loss'])


if __name__ == '__main__':
    run(20, 10)
