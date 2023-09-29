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

# Define the model
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(1,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model (You can use your own dataset for training)
# For simplicity, we'll skip training in this example

# Input from the user
user_input = input("Enter a number to check if it's prime: ")
test_number = np.array([int(user_input)])

# Load pre-trained model weights (you can train the model earlier)
# model.load_weights('your_model_weights.h5')

# Example of using the trained model to predict if a number is prime
predicted_result = model.predict(test_number)

print(f"The model predicts that {test_number[0]} is {'prime' if predicted_result[0][0] > 0.5 else 'not prime'}.")