from keras import models, applications, layers


def vgg_model():
    model = models.Sequential()
    base_conv = applications.VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    base_conv.trainable = False
    model.add(base_conv)
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model
