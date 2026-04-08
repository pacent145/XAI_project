import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=3)

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

# Show one image
plt.imshow(x_test[10], cmap='gray')
plt.title("Test Image")
plt.show()

# Predict
prediction = model.predict(x_test)
print("Predicted digit:", prediction[10].argmax())
# SHAP
import shap
import numpy as np

# Select small background sample for SHAP
background = x_train[:100]

# Create SHAP explainer
explainer = shap.DeepExplainer(model, background)

# Explain one image
test_image = x_test[10:11]

shap_values = explainer.shap_values(test_image)

# Plot explanation
shap.image_plot(shap_values, test_image)