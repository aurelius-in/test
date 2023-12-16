import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = 'label_scores.csv'  # Update the path if necessary
data = pd.read_csv(file_path)

# Exclude the 'PRCP ID' column and any other non-feature columns
features = data.drop(columns=['PRCP ID'])  # Adjust if there are other non-feature columns

# Assuming the target variable (scores) is one of the columns in the CSV
# Replace 'target_column_name' with the actual name of your target column
X = features.drop(columns=['target_column_name']).values
y = features['target_column_name'].values

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# VAE Model
class VAE(Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(X_train.shape[1],)),
            layers.Dense(64, activation='relu'),
            layers.Dense(latent_dim * 2),
        ])

        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(X_train.shape[1], activation='sigmoid')
        ])

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def call(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decode(z)
        return x_recon, mean, logvar

# Loss function for VAE
def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

def compute_loss(model, x):
    x_recon, mean, logvar = model(x)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_recon, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1])
    logpz = log_normal_pdf(mean, 0., 0.)
    logqz_x = log_normal_pdf(mean, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

# Training the VAE
epochs = 10  # Set the number of epochs
latent_dim = 10  # Set the latent dimension
vae = VAE(latent_dim)
optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training loop
for epoch in range(1, epochs + 1):
    for train_x in np.array_split(X_train, 10):  # Batch training
        loss = train_step(vae, train_x, optimizer)
    print(f'Epoch {epoch}, Loss: {loss.numpy()}')

# Feature Extraction using VAE
means, _ = vae.encode(X_train)
reduced_features = vae.reparameterize(means, np.zeros_like(means))

# Train a scoring model
scoring_model = LinearRegression()
scoring_model.fit(reduced_features, y_train)

# Calculate Status Scores on test data
test_means, _ = vae.encode(X_test)
test_reduced_features = vae.reparameterize(test_means, np.zeros_like(test_means))
status_scores = scoring_model.predict(test_reduced_features)

print(status_scores)  # Output the Status
