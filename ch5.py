from keras import layers, models, optimizers, utils, preprocessing, applications
import matplotlib.pyplot as plt
import os
import shutil


def create_small_dataset():
    original_dataset_dir = 'cats_vs_dogs_data'

    base_dir = 'cats_and_dogs_small'
    os.mkdir(base_dir)

    train_dir = os.path.join(base_dir, 'train')
    os.mkdir(train_dir)
    validation_dir = os.path.join(base_dir, 'validation')
    os.mkdir(validation_dir)
    test_dir = os.path.join(base_dir, 'test')
    os.mkdir(test_dir)

    train_cats_dir = os.path.join(train_dir, 'cats')
    os.mkdir(train_cats_dir)

    train_dogs_dir = os.path.join(train_dir, 'dogs')
    os.mkdir(train_dogs_dir)

    validation_cats_dir = os.path.join(validation_dir, 'cats')
    os.mkdir(validation_cats_dir)

    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    os.mkdir(validation_dogs_dir)

    test_cats_dir = os.path.join(test_dir, 'cats')
    os.mkdir(test_cats_dir)

    test_dogs_dir = os.path.join(test_dir, 'dogs')
    os.mkdir(test_dogs_dir)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, 'train', fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, 'train', fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, 'train', fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, 'train', fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, 'train', fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, 'train', fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)


def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                            input_shape=(150, 150, 3)))  # (148, 148)
    model.add(layers.MaxPool2D((2, 2)))  # (74, 74)
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))  # 72, 72
    model.add(layers.MaxPool2D((2, 2)))  # 36, 36
    model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))  # 34, 34
    model.add(layers.MaxPool2D((2, 2)))  # 17, 17
    model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))  # 15, 15
    model.add(layers.MaxPool2D((2, 2)))  # 7, 7

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


def run_model(model, steps_per_epoch=80, epochs=80):
    model.compile(
        optimizer=optimizers.RMSprop(lr=1e-4),
        loss='binary_crossentropy',
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

    model.save('cats_vs_dogs.h5')
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
        'cats_vs_dogs_small/train/', target_size=(150, 150), batch_size=20, class_mode='binary')

    valid_gen = valid_data_gen.flow_from_directory(
        'cats_vs_dogs_small/validation/',
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

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
    base_conv = applications.VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    base_conv.trainable = False
    model.add(base_conv)
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


if __name__ == '__main__':
    # create_small_dataset()
    # model = build_model()
    model = vgg_model()
    print(model.summary())

    model, history = run_model(model)
    plot_training(history)
