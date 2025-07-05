import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle

# Load data
X = np.load("gesture_data.npy")
y = np.load("gesture_labels.npy")

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = SVC(kernel='rbf', probability=True)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("[INFO] Classification Report:\n", classification_report(y_test, y_pred, target_names=encoder.classes_))

# Save model and encoder
with open("gesture_model.pkl", "wb") as f:
    pickle.dump((model, encoder), f)

print("[INFO] Model and encoder saved.")
