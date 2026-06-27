import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Multiply

def residual_block(x, filters):
    shortcut = x
    
    # First Conv
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Second Conv
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    
    # Match dimensions for shortcut if needed
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
        
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def gated_attention_layer(x):
    # A simple spatial attention mechanism
    # We compute attention weights over the spatial dimensions
    attention = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)
    # Multiply the input feature map by the attention weights
    x = Multiply()([x, attention])
    return x

def build_cgface(input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape)
    
    # Initial Convolution
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    # Residual Blocks
    x = residual_block(x, 64)
    x = residual_block(x, 128)
    x = residual_block(x, 256)
    
    # Gated Attention Mechanism
    x = gated_attention_layer(x)
    
    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)
    
    # Dense Layers
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    
    # Output Layer
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model
