import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Define the CNN model architecture
def create_model(learning_rate, batch_size, num_filters, num_layers):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    for _ in range(num_layers - 1):
        model.add(tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
# Define the objective function to evaluate the model
def objective_function(params):
    learning_rate, batch_size, num_filters, num_layers = params
    model = create_model(learning_rate, int(batch_size), int(num_filters), int(num_layers))
    model.fit(X_train, y_train, batch_size=int(batch_size), epochs=5, verbose=0)
    y_pred_prob = model.predict(X_val)
    y_pred = np.argmax(y_pred_prob, axis=1)
    accuracy = accuracy_score(y_val, y_pred)
    return -accuracy  # maximize accuracy


# PSO implementation
def pso_search(num_particles, num_iterations):
    particles = np.random.uniform(low=0, high=1, size=(num_particles, 4))
    velocities = np.zeros_like(particles)
    global_best = None
    global_best_fitness = np.inf

    for _ in range(num_iterations):
        for i in range(num_particles):
            # Ensure num_filters is at least 1
            num_filters = max(1, particles[i][2])
            particles[i][2] = num_filters

            fitness = objective_function(particles[i])
            if fitness < global_best_fitness:
                global_best = particles[i].copy()
                global_best_fitness = fitness
            best_particle = particles[i].copy()

            velocities[i] = 0.5 * velocities[i] + 2 * np.random.random() * (best_particle - particles[i]) \
                            + 2 * np.random.random() * (global_best - particles[i])
            particles[i] += velocities[i]

            particles[i] = np.clip(particles[i], 0, 1)

    return global_best

# Load and preprocess the dataset
(X_train_full, y_train_full), (_, _) = mnist.load_data()
X_train_full = X_train_full.reshape((-1, 28, 28, 1)) / 255.0
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Select a subset of the data for faster training
subset_size = 10000
subset_indices = np.random.choice(X_train.shape[0], size=subset_size, replace=False)
X_train_subset = X_train[subset_indices]
y_train_subset = y_train[subset_indices]






# Set the hyperparameters for PSO
num_particles = 20
num_iterations = 10

# Perform PSO search for hyperparameter optimization
best_hyperparameters = pso_search(num_particles, num_iterations)


# Train the CNN model with the best hyperparameters
learning_rate, batch_size, num_filters, num_layers = best_hyperparameters
best_model = create_model(learning_rate, int(batch_size), int(num_filters), int(num_layers))

num_epochs = 5
new_batch_size = 16  # Decrease the batch size

for epoch in range(num_epochs):
    best_model.fit(X_train, y_train, batch_size=new_batch_size, epochs=1, verbose=0)  # Train for one epoch
    loss, accuracy = best_model.evaluate(X_test, y_test)  # Evaluate on the test set
    print("Epoch:", epoch+1, "- Test Loss:", loss, "- Test Accuracy:", accuracy)