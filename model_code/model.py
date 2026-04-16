"""
model_code/model.py
====================
FIR thermal dataset model definitions.
Three architectures available: DW-CNN2D, 3D-CNN, CNN-LSTM.

FIX: Updated imports to use tensorflow.keras (TF 2.x / Keras 3.x compatible).
     ModelCheckpoint now saves in .keras format (Keras 3.x standard).

NOTE: This file is only needed for training on the FIR thermal dataset.
      The live inference pipeline (run_inference.py) does NOT use this model —
      it uses MediaPipe Pose instead.
"""

import matplotlib.pyplot as plt

try:
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        DepthwiseConv2D, BatchNormalization, Activation, MaxPooling2D,
        Dropout, Flatten, Dense, Conv3D, MaxPool3D, TimeDistributed,
        Conv2D, LSTM
    )
    from tensorflow.keras.losses import CategoricalCrossentropy
except ImportError:
    raise ImportError(
        "TensorFlow is required for model training.\n"
        "Install it with: pip install tensorflow>=2.13.0\n"
        "(This is NOT required for live inference with run_inference.py)"
    )


def model_dwcnn2d(frame_num: int = 8):
    """
    Depthwise-separable 2D-CNN operating on (frame_num, 32, 24) thermal sequences.
    data_format='channels_first' treats frame_num as the channel dimension.
    """
    model = Sequential()
    model.add(DepthwiseConv2D(kernel_size=3, input_shape=(frame_num, 32, 24),
                              padding='same', data_format='channels_first'))
    model.add(DepthwiseConv2D(kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
    model.add(Dropout(0.25))

    model.add(DepthwiseConv2D(kernel_size=3, padding='same'))
    model.add(DepthwiseConv2D(kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    return model


def model_3dcnn(frame_num: int = 8):
    """3D-CNN operating on (32, 24, frame_num, 1) thermal volume."""
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3),
                     input_shape=(32, 24, frame_num, 1), padding='same'))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool3D(pool_size=(2, 2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same'))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool3D(pool_size=(2, 2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    return model


def model_cnn_lstm(frame_num: int = 8):
    """CNN-LSTM: spatial features per frame + temporal LSTM."""
    model = Sequential()
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu'),
                              input_shape=(frame_num, 32, 24, 1)))
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(2, 2)))
    model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu')))
    model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(2, 2)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Flatten()))
    model.add(Dropout(0.2))

    model.add(LSTM(32, return_sequences=False, dropout=0.2))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    return model


def train(model, train_generator, valid_generator):
    """
    Train the model.

    Parameters
    ----------
    model : keras.Model
    train_generator : data generator with .n and .batch_size attributes
    valid_generator : data generator with .n and .batch_size attributes
    """
    model.compile(
        loss=CategoricalCrossentropy(),
        optimizer=Adam(),
        metrics=['accuracy']
    )
    # FIX: use .keras extension (Keras 3.x standard; .h5 is legacy)
    checkpoint = ModelCheckpoint(
        filepath='models/model_{epoch:02d}_{val_loss:.4f}.keras',
        monitor='val_loss',
        mode='auto',
        save_best_only=True
    )

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

    history = model.fit(
        train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        epochs=100,
        validation_data=valid_generator,
        validation_steps=STEP_SIZE_VALID,
        callbacks=[checkpoint]
    )
    return history


def show_train_result(history):
    history_dict    = history.history

    loss_values     = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc_values      = history_dict['accuracy']
    val_acc_values  = history_dict['val_accuracy']

    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, loss_values,     'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b',  label='Validation loss')
    plt.plot(epochs, acc_values,      'go', label='Training acc')
    plt.plot(epochs, val_acc_values,  'g',  label='Validation acc')
    plt.title('Training and Validation Loss / Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Accuracy')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Uncomment the model you want to test:
    # model = model_cnn_lstm()
    # model = model_3dcnn()
    model = model_dwcnn2d()
    # Provide real generators to actually train:
    # history = train(model, train_generator, valid_generator)
    # show_train_result(history)
