import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from models.dcgan import build_generator, build_discriminator
import numpy as np
import os
import glob
import cv2

def load_gan_data(img_size=(84, 84)):
    # Load only REAL images for GAN training
    real_paths = glob.glob('data/processed_frames/REAL/*.jpg')
    X = []
    
    for p in real_paths:
        img = cv2.imread(p)
        if img is not None:
            img = cv2.resize(img, img_size)
            X.append(img)
            
    X = np.array(X, dtype=np.float32)
    # Normalize to [-1, 1] for Tanh
    X = (X - 127.5) / 127.5 
    return X

def train():
    X_train = load_gan_data()
    if len(X_train) == 0:
        print("No REAL data found for GAN.")
        return
        
    print(f"Loaded {len(X_train)} samples for GAN.")
    
    noise_dim = 100
    batch_size = 64
    epochs = 50
    
    generator = build_generator(noise_dim)
    discriminator = build_discriminator()
    
    cross_entropy = BinaryCrossentropy(from_logits=False)
    
    gen_optimizer = Adam(2e-4, beta_1=0.5)
    disc_optimizer = Adam(2e-4, beta_1=0.5)
    
    @tf.function
    def train_step(images):
        noise = tf.random.normal([batch_size, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
            
            real_loss = cross_entropy(tf.ones_like(real_output), real_output)
            fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
            disc_loss = real_loss + fake_loss

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        
        return gen_loss, disc_loss

    # Training Loop
    dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(len(X_train)).batch(batch_size, drop_remainder=True)
    
    print("Training DCGAN...")
    for epoch in range(epochs):
        gen_loss_avg = 0
        disc_loss_avg = 0
        steps = 0
        for image_batch in dataset:
            g_loss, d_loss = train_step(image_batch)
            gen_loss_avg += g_loss
            disc_loss_avg += d_loss
            steps += 1
            
        print(f"Epoch {epoch+1}/{epochs} | Gen Loss: {gen_loss_avg/steps:.4f} | Disc Loss: {disc_loss_avg/steps:.4f}")
        
    os.makedirs('saved_models', exist_ok=True)
    generator.save('saved_models/dcgan_generator.h5')
    discriminator.save('saved_models/dcgan_discriminator.h5')
    print("DCGAN models saved.")

if __name__ == '__main__':
    train()
