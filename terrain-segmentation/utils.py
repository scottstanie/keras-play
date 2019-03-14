from keras import preprocessing


def preproc_data_with_masks(batch_size=16, target_size=(256, 256)):
    """Returns: zip(image_gen, mask_gen)"""
    data_gen_args = dict(
        rescale=1.0 / 255.,
        rotation_range=40,
        width_shift_range=.2,  # randomly translate image by .2 of total width
        height_shift_range=.2,
        shear_range=.2,
        zoom_range=.5,
        horizontal_flip=True,
        fill_mode='nearest',
    )
    image_data_gen = preprocessing.image.ImageDataGenerator(**data_gen_args)
    mask_data_gen = preprocessing.image.ImageDataGenerator(**data_gen_args)
    seed = 1  # Makes sure the randomness matches images/masks
    # image_data_gen.fit()  # Only if zca_whiteneing or stdnorm

    image_gen = image_data_gen.flow_from_directory(
        'data/images/train',
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None,
        seed=seed)

    mask_gen = mask_data_gen.flow_from_directory(
        'data/labels/train',
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None,
        seed=seed)

    valid_data_gen = preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.)
    valid_data_mask_gen = preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.)

    valid_gen = valid_data_gen.flow_from_directory(
        'data/images/val',
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None,
        seed=seed)

    valid_mask_gen = valid_data_mask_gen.flow_from_directory(
        'data/labels/val',
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None,
        seed=seed)

    return zip(image_gen, mask_gen), zip(valid_gen, valid_mask_gen)
