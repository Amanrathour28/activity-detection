import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.metrics import Accuracy
from keras.models import Sequential
from keras.layers import (
    DepthwiseConv2D, BatchNormalization, Activation, MaxPooling2D, MaxPool2D,
    Dropout, Flatten, Dense, Conv3D, MaxPool3D, TimeDistributed,
    Conv2D, LSTM
)
from keras.losses import CategoricalCrossentropy


def model_dwcnn2d(frame_num: int = 8):
    model = Sequential()
    model.add(DepthwiseConv2D(kernel_size=3, input_shape=(frame_num, 32, 24), padding='same', data_format='channels_first'))
    model.add(DepthwiseConv2D(kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(DepthwiseConv2D(kernel_size=3, padding='same'))
    model.add(DepthwiseConv2D(kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    return model


def model_3dcnn(frame_num: int = 8):
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(32, 24, frame_num, 1), padding='same'))
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
    model = Sequential()
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu'), input_shape=(frame_num, 32, 24, 1)))
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
    checkpoint = ModelCheckpoint(
        filepath='models/model_{epoch:02d}_{val_loss:.4f}.h5',
        monitor='val_loss',
        mode='auto',
        save_best_only=True
    )

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

    # NOTE: model.fit() replaces the deprecated model.fit_generator()
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
    history_dict = history.history   # FIX: was referenced but never assigned

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
    # model = model_cnn_lstm()
    # model = model_3dcnn()
    model = model_dwcnn2d()
    # Provide real generators to actually train:
    # history = train(model, train_generator, valid_generator)
    # show_train_result(history)
