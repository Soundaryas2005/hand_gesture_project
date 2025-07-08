import os
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from food_model import build_model
from sklearn.preprocessing import LabelEncoder

# === Step 1: Define labels (must match training order) ===
labels = ['Apple', 'Pizza', 'Burger']  # Example classes
le = LabelEncoder()
le.fit(labels)

# === Step 2: Load model weights ===
model_path = "food_model.h5"

if not os.path.exists(model_path):
    print(f"[ERROR] Model file '{model_path}' not found.")
    print("Please train the model first by running: python train.py")
    exit()

# Build model and load weights
model = build_model(num_classes=len(labels))
model.load_weights(model_path)

# === Step 3: Prediction function ===
def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"[ERROR] Image file '{img_path}' not found.")
        return None, None

    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224)) / 255.0
    x = np.expand_dims(img, axis=0)

    pred_class, pred_cal = model.predict(x)
    food = le.inverse_transform([np.argmax(pred_class)])
    return food[0], round(pred_cal[0][0], 2)

# === Step 4: Run example prediction ===
test_img = "dataset/images/pizza.jpg"
food, cal = predict_image(test_img)

if food:
    print(f"✅ Detected: {food} | 🔥 Estimated Calories: {cal} kcal")

    # Save result
    with open("prediction_result.txt", "w") as f:
        f.write(f"Detected: {food} | Estimated Calories: {cal} kcal\n")
