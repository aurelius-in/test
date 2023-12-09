!{sys.executable} -m pip install tensorflow numpy pandas scikit-learn matplotlib

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def build_vae(input_dim, latent_dim=2):
    # Encoder
    inputs = Input(shape=(input_dim,), name='encoder_input')
    x = Dense(128, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # Use reparameterization trick to ensure correct gradient
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # Instantiate encoder
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # Decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(128, activation='relu')(latent_inputs)
    outputs = Dense(input_dim, activation='sigmoid')(x)

    # Instantiate decoder
    decoder = Model(latent_inputs, outputs, name='decoder')

    # VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')

    # VAE loss
    reconstruction_loss = binary_crossentropy(inputs, outputs)
    reconstruction_loss *= input_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    return vae, encoder, decoder

# Example usage
vae, encoder, decoder = build_vae(input_dim=30)  # replace 30 with the number of features in your dataset

#


import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
synthetic_data = pd.read_csv('path_to_synthetic_data.csv')
real_data = pd.read_csv('path_to_real_data.csv')

# Preprocess data
scaler = StandardScaler()
X_synthetic = scaler.fit_transform(synthetic_data.drop('target', axis=1))
y_synthetic = synthetic_data['target']
X_real = scaler.transform(real_data.drop('target', axis=1))
y_real = real_data['target']

#

vae.compile(optimizer='adam')
vae.fit(X_synthetic, epochs=100, batch_size=32)  # Adjust epochs and batch_size as needed

#

import numpy as np
from sklearn.metrics import mean_squared_error

best_features = []
best_loss = float('inf')

for feature in range(X_synthetic.shape[1]):
    current_features = best_features + [feature]
    vae, _, _ = build_vae(len(current_features))
    vae.compile(optimizer='adam')
    vae.fit(X_synthetic[:, current_features], epochs=100, batch_size=32, verbose=0)

    reconstructed = vae.predict(X_real[:, current_features])
    loss = mean_squared_error(X_real[:, current_features], reconstructed)

    if loss < best_loss:
        best_loss = loss
        best_features.append(feature)
        print(f"Feature {feature} added, New loss: {loss}")

print(f"Best features: {best_features}")

