import os
from keras import layers, models, optimizers, preprocessing, applications
import matplotlib.pyplot as plt
from prep_images import split_data

NUM_CLASSES = 25


def run_model(model, steps_per_epoch=80, epochs=80):
    model.compile(
        optimizer=optimizers.RMSprop(lr=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    train_gen, valid_gen = preproc_data()
    validation_steps = epochs // 2
    history = model.fit_generator(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=valid_gen,
        validation_steps=validation_steps,
    )

    model.save('msl-image-model-vgg.h5')
    return model, history


def preproc_data():
    train_data_gen = preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.,
        rotation_range=40,
        width_shift_range=.2,  # randomly translate image by .2 of total width
        height_shift_range=.2,
        shear_range=.2,
        zoom_range=.2,
        horizontal_flip=True,
        fill_mode='nearest',
    )

    valid_data_gen = preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.)

    train_gen = train_data_gen.flow_from_directory(
        'msl-images/train/', target_size=(128, 128), batch_size=10, class_mode='categorical')

    valid_gen = valid_data_gen.flow_from_directory(
        'msl-images/val/', target_size=(128, 128), batch_size=10, class_mode='categorical')

    return train_gen, valid_gen


def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def vgg_model():
    model = models.Sequential()
    base_conv = applications.VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_conv.trainable = False
    model.add(base_conv)
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
    return model


if __name__ == '__main__':

    if not os.path.exists('msl-images/train/'):
        split_data()

    model = vgg_model()
    print(model.summary())

    model, history = run_model(model)
    plot_training(history)
