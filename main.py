import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import shap

# Load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=5)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

# Pick random image
random_index = np.random.randint(0, len(x_test))

# Show image
plt.imshow(x_test[random_index], cmap='gray')
plt.title(f"Test Image (Index: {random_index})")
plt.axis('off')
plt.show()

# Predict only that image
prediction = model.predict(x_test[random_index:random_index+1])

predicted_digit = np.argmax(prediction[0])
confidence = np.max(prediction[0]) * 100

print("Predicted digit:", predicted_digit)
print("Actual digit:", y_test[random_index])
print(f"Confidence: {confidence:.2f}%")

# -------- SHAP (SAFE VERSION) --------

# Reduce background for speed
background = x_train[:50]

# Create explainer
explainer = shap.DeepExplainer(model, background)

# Explain only ONE image
test_image = x_test[random_index:random_index+1]

shap_values = explainer.shap_values(test_image)

# Plot SHAP explanation
shap.image_plot(shap_values, test_image)