from segmentation_models import Unet
from keras.callbacks import ModelCheckpoint  # , EarlyStopping
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

import utils

# BACKBONE = 'resnet34'
BACKBONE = 'vgg16'
preprocess_input = get_preprocessing(BACKBONE)

BATCH_SIZE = 8
TARGET_SIZE = (224, 224)
# x_train, y_train, x_val, y_val = load_data(...)

# image_gen, mask_gen = train_gen
train_gen, valid_gen = utils.preproc_data_with_masks(BATCH_SIZE, TARGET_SIZE)

# If loading actual numpy arrays, need:
# x_val = preprocess_input(x_val)

# define model
model = Unet(
    BACKBONE,
    encoder_weights='imagenet',
    classes=1,
    activation='sigmoid',
    encoder_freeze=True,
)
model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])
# model.compile('Adadelta', loss='binary_crossentropy')
print(model.summary())

callbacks = [
    ModelCheckpoint('model_weights.h5', monitor='val_loss', save_best_only=True, verbose=0)
]

# fit model
model.fit_generator(
    train_gen,
    steps_per_epoch=80,
    epochs=50,
    callbacks=callbacks,
    validation_data=valid_gen,
)
model.save("unet.h5")
