import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load your saved generator model
generator = tf.keras.models.load_model('./mnist_generator.h5')

# Load a pre-trained MNIST digit classifier (simple CNN)
# This classifier is used only to filter generated images by predicted digit
mnist_classifier = tf.keras.models.load_model('https://tfhub.dev/tensorflow/tfjs-model/mnist_digit_classifier/1', compile=False)
# (If you don't have a classifier, you can train one or download a simple MNIST classifier.)

def generate_images_for_digit(generator, target_digit, noise_dim=100, max_attempts=1000, images_needed=5):
    selected_images = []
    attempts = 0
    while len(selected_images) < images_needed and attempts < max_attempts:
        noise = tf.random.normal([1, noise_dim])
        generated_img = generator(noise, training=False)
        # Denormalize from [-1,1] to [0,1]
        img = (generated_img * 0.5 + 0.5).numpy()
        # Classify generated image to get predicted digit
        pred = np.argmax(mnist_classifier.predict(img), axis=1)[0]
        if pred == target_digit:
            selected_images.append(img[0, :, :, 0])
        attempts += 1
    return selected_images

st.title("MNIST Digit Generator GAN Demo")

digit = st.slider("Choose a digit to generate:", 0, 9, 0)
if st.button("Generate 5 images"):
    st.write(f"Generating images of digit {digit}... This may take a moment.")
    images = generate_images_for_digit(generator, digit)
    if len(images) == 0:
        st.write("Sorry, could not generate the requested digit this time. Try again!")
    else:
        cols = st.columns(5)
        for idx, img in enumerate(images):
            with cols[idx]:
                st.image(img, width=80, clamp=True)
