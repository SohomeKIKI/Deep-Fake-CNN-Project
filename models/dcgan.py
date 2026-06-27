import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, BatchNormalization, LeakyReLU, Conv2D, Dropout, Flatten, Input

def build_generator(noise_dim=100):
    model = Sequential([
        Input(shape=(noise_dim,)),
        Dense(7 * 7 * 256, use_bias=False),
        BatchNormalization(),
        LeakyReLU(),
        Reshape((7, 7, 256)),
        
        # 7x7 -> 14x14
        Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),
        
        # 14x14 -> 28x28
        Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),
        
        # 28x28 -> 84x84 (stride 3)
        Conv2DTranspose(32, (5, 5), strides=(3, 3), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),
        
        # Output layer to map to 3 channels, 84x84
        Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh')
    ])
    return model

def build_discriminator(image_shape=(84, 84, 3)):
    model = Sequential([
        Input(shape=image_shape),
        Conv2D(64, (5, 5), strides=(2, 2), padding='same'),
        LeakyReLU(),
        Dropout(0.3),
        
        Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        LeakyReLU(),
        Dropout(0.3),
        
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    return model
