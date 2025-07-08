import pandas as pd
import numpy as np
import os
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from food_model import build_model
from tensorflow.keras.callbacks import ModelCheckpoint

# === Load labels and image paths ===
df = pd.read_csv("dataset/labels.csv")
df['image_path'] = df['image_name'].apply(lambda x: os.path.join('dataset/images', x))

# Encode food labels
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['food_label'])

# === Load and preprocess images ===
X = []
for path in df['image_path']:
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    X.append(img / 255.0)
X = np.array(X)

# === Targets ===
y_class = to_categorical(df['label_encoded'])
y_calories = df['calories'].values

# === Split dataset ===
X_train, X_val, y_class_train, y_class_val, y_cal_train, y_cal_val = train_test_split(
    X, y_class, y_calories, test_size=0.2, random_state=42)

# === Data augmentation ===
aug = ImageDataGenerator(rotation_range=10, zoom_range=0.1, horizontal_flip=True)

# === Build and compile model ===
model = build_model(num_classes=y_class.shape[1])
model.compile(optimizer='adam',
              loss={'food_class': 'categorical_crossentropy', 'calories': 'mse'},
              metrics={'food_class': 'accuracy', 'calories': 'mae'})

# === Save best model ===
checkpoint = ModelCheckpoint("food_model.h5", monitor='val_food_class_accuracy', save_best_only=True)

# === Train ===
model.fit(
    aug.flow(X_train, {'food_class': y_class_train, 'calories': y_cal_train}, batch_size=32),
    validation_data=(X_val, {'food_class': y_class_val, 'calories': y_cal_val}),
    epochs=10,
    callbacks=[checkpoint]
)
