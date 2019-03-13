from keras import layers, models


def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(64, kernel_size=(3, 3), input_size=(150, 150, 3)))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(32, kernel_size=(3, 3)))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(32, kernel_size=(3, 3)))
    model.add(layers.MaxPool2D((2, 2)))

    return model
