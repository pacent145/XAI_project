import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import cv2
import psutil
import time
import os
import json
from PIL import Image

os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/heatmaps", exist_ok=True)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train / 255.0
x_test  = x_test  / 255.0
x_train = x_train[..., np.newaxis]
x_test  = x_test[..., np.newaxis]

model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv1'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv3'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

cpu_before = psutil.cpu_percent(interval=1)
mem_before = psutil.virtual_memory().used / (1024 ** 2)
start_time = time.time()

history = model.fit(x_train, y_train, epochs=5, validation_split=0.1, verbose=1)

training_time = time.time() - start_time
cpu_after  = psutil.cpu_percent(interval=1)
mem_after  = psutil.virtual_memory().used / (1024 ** 2)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

metrics = {
    "test_accuracy": round(float(test_acc) * 100, 2),
    "test_loss": round(float(test_loss), 4),
    "training_time_seconds": round(training_time, 1),
    "cpu_before": cpu_before,
    "cpu_after": cpu_after,
    "memory_increase_mb": round(mem_after - mem_before, 1),
    "train_accuracy": [round(float(v) * 100, 2) for v in history.history['accuracy']],
    "val_accuracy":   [round(float(v) * 100, 2) for v in history.history['val_accuracy']],
    "train_loss":     [round(float(v), 4) for v in history.history['loss']],
    "val_loss":       [round(float(v), 4) for v in history.history['val_loss']],
}

with open("outputs/metrics.json", "w") as f:
    json.dump(metrics, f)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
epochs_range = range(1, 6)
axes[0].plot(epochs_range, metrics['train_accuracy'], 'b-o', label='Train Accuracy')
axes[0].plot(epochs_range, metrics['val_accuracy'],   'r-o', label='Val Accuracy')
axes[0].set_title('Model Accuracy per Epoch')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy (%)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, metrics['train_loss'], 'b-o', label='Train Loss')
axes[1].plot(epochs_range, metrics['val_loss'],   'r-o', label='Val Loss')
axes[1].set_title('Model Loss per Epoch')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Training Performance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/training_curves.png", dpi=150, bbox_inches='tight')
plt.close()

model.save("outputs/prescription_model.keras")
print(f"\nModel saved. Test Accuracy: {metrics['test_accuracy']}%")
print(f"Training Time: {metrics['training_time_seconds']}s | Memory Increase: {metrics['memory_increase_mb']} MB")


def make_gradcam(img_array, model, layer_name='conv3'):
    _ = model(img_array)
    grad_model = keras.models.Model(
        inputs=model.inputs,
       outputs=[model.get_layer(layer_name).output, model.outputs[0]]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        class_idx = tf.argmax(preds[0])
        score = preds[:, class_idx]
    grads = tape.gradient(score, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def load_image(path):
    img = Image.open(path).convert("L")
    img = img.resize((28, 28))
    arr = np.array(img) / 255.0
    arr = arr[..., np.newaxis]
    arr = np.expand_dims(arr, axis=0)
    return arr


def authenticate(image_path, threshold=85.0):
    img_array = load_image(image_path)

    infer_start = time.time()
    preds = model.predict(img_array, verbose=0)
    infer_time = (time.time() - infer_start) * 1000

    predicted_digit = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0])) * 100
    status = "AUTHENTIC" if confidence >= threshold else "SUSPICIOUS"

    heatmap = make_gradcam(img_array, model)
    heatmap_resized = cv2.resize(heatmap, (28, 28))
    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
    original_rgb = np.repeat(img_array[0], 3, axis=2)
    overlay = np.clip(0.4 * heatmap_colored + 0.6 * original_rgb, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor('#0f0f1a')

    axes[0].imshow(img_array[0, :, :, 0], cmap='gray')
    axes[0].set_title('Input Image', color='white', fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap\n(Red = model focused here)', color='white', fontweight='bold')
    axes[1].axis('off')

    color = '#00ff88' if status == "AUTHENTIC" else '#ff4444'
    axes[2].imshow(overlay)
    axes[2].set_title(f'Result: {status}\nDigit: {predicted_digit} | Confidence: {confidence:.1f}%',
                      color=color, fontweight='bold')
    axes[2].axis('off')

    for ax in axes:
        ax.set_facecolor('#0f0f1a')

    plt.suptitle('Prescription Digit Authentication System', color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()

    save_name = f"outputs/heatmaps/result_{predicted_digit}_{int(confidence)}.png"
    plt.savefig(save_name, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

    result = {
        "predicted_digit": predicted_digit,
        "confidence": round(confidence, 2),
        "status": status,
        "inference_time_ms": round(infer_time, 2),
        "heatmap_path": save_name,
        "all_probabilities": {str(i): round(float(preds[0][i]) * 100, 2) for i in range(10)}
    }

    with open("outputs/last_result.json", "w") as f:
        json.dump(result, f)

    return result


random_index = np.random.randint(0, len(x_test))
test_img = x_test[random_index]
img_pil = Image.fromarray((test_img[:, :, 0] * 255).astype(np.uint8))
temp_path = "outputs/temp_test.png"
img_pil.save(temp_path)

result = authenticate(temp_path)
print(f"\nPredicted: {result['predicted_digit']} | Actual: {y_test[random_index]}")
print(f"Confidence: {result['confidence']}% | Status: {result['status']}")
print(f"Inference Time: {result['inference_time_ms']} ms")
print(f"Heatmap saved to: {result['heatmap_path']}")