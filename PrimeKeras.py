import numpy as np
import tensorflow as tf
from tensorflow import keras

# Function to check if a number is prime
def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

# Generate a list of numbers and their corresponding labels
num_samples = 1000
max_number = 10000

numbers = np.random.randint(2, max_number, num_samples)
labels = [1 if is_prime(num) else 0 for num in numbers]

# Convert numbers and labels to numpy arrays
X = np.array(numbers)
y = np.array(labels)

# Define the model
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(1,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# Example of using the trained model to predict if a number is prime
test_number = np.array([17])  # Change this number to test different values
predicted_result = model.predict(test_number)

print(f"The model predicts that {test_number[0]} is {'prime' if predicted_result[0][0] > 0.5 else 'not prime'}.")