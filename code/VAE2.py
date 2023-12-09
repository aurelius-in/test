# After loading and preprocessing your data
num_features = X_synthetic.shape[1]  # Determine the number of features in the synthetic data

# Initialize VAE with the correct input dimension
vae, _, _ = build_vae(input_dim=num_features)

# Compile and fit the VAE
vae.compile(optimizer='adam')
vae.fit(X_synthetic, epochs=1000, batch_size=64)
