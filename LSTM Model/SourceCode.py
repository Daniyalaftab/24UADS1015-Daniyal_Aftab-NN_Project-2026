import pandas as pd
import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import load_model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical           
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ================= PATH SETTINGS =================
TRAINING_FOLDERS = {
    0: r"C:\FinalDataset\dataset\dataset\1 star",
    1: r"C:\FinalDataset\dataset\dataset\2 star",
    2: r"C:\FinalDataset\dataset\dataset\3 star",
    3: r"C:\FinalDataset\dataset\dataset\4 star",
    4: r"C:\FinalDataset\dataset\dataset\5 star"
}

MODEL_SAVE_LSTM = r"C:\Anti\lstm_model13.h5"
SCALER_SAVE = r"C:\Anti\scaler13.pkl"
TEST_FILE = r"C:\Sample1data\1 star\Daniyal_aggresive2.csv"

# ================= LOSS SETTINGS =================
WINDOW_SIZE = 120
STEP_SIZE = 20

FEATURES = ['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z','acc_mag','gyro_mag']
LABEL_NAMES = ['Aggressive', 'Rush', 'Moderate', 'Smooth', 'Safe']

# ================= LOAD+WINDOW DATA =================
print("\n[1] Loading Dataset...")

X, y = [], []

for label, folder in TRAINING_FOLDERS.items():
    print(f"Processing: {folder}")

    csv_files = glob.glob(os.path.join(folder, "*.csv"))

    for file in csv_files:
        try:
            df = pd.read_csv(file)

            # Fix column names
            if 'X_Acc' in df.columns:
                df.rename(columns={
                    'X_Acc': 'acc_x', 'Y_Acc': 'acc_y', 'Z_Acc': 'acc_z',
                    'X_Gyro': 'gyro_x', 'Y_Gyro': 'gyro_y', 'Z_Gyro': 'gyro_z'
                }, inplace=True)

            # Safety check
            if not all(col in df.columns for col in ['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']):
                print(f"Skipping {file}")
                continue

            #  Feature Engineering
            df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
            df['gyro_mag'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)

            values = df[FEATURES].values.astype(np.float32)

            for i in range(0, len(values) - WINDOW_SIZE, STEP_SIZE):
                X.append(values[i:i + WINDOW_SIZE])
                y.append(label)

        except Exception as e:
            print(f"Error in {file}: {e}")

X = np.array(X)
y = np.array(y)

print("Dataset Shape:", X.shape)


# ================= Split=================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))

print("\nClass Weights:", class_weights)


# ================= Scaling =================
scaler = StandardScaler()
scaler.fit(X_train.reshape(-1, len(FEATURES)))

X_train = scaler.transform(X_train.reshape(-1, len(FEATURES))).reshape(len(X_train), WINDOW_SIZE, len(FEATURES))
X_test = scaler.transform(X_test.reshape(-1, len(FEATURES))).reshape(len(X_test), WINDOW_SIZE, len(FEATURES))

with open(SCALER_SAVE, 'wb') as f:  
    pickle.dump(scaler, f)


# ============================================================
# MODEL BUILDING (CNN + BIDIRECTIONAL LSTM)
# ============================================================
from tensorflow.keras.layers import Conv1D, MaxPooling1D

lstm_model= Sequential([
    # CNN PART (detects local patterns)
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(WINDOW_SIZE, len(FEATURES))),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    
    # LSTM PART (sequence learning)
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    
    # DENSE PART
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(5, activation='softmax')
])

lstm_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

lstm_model.summary()



# ================= Trainning =================
print("\n[2] Training LSTM File...")
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
)

y_train_final = to_categorical(y_train_final, 5)
y_val = to_categorical(y_val, 5)
y_test_cat = to_categorical(y_test, 5)

history = lstm_model.fit(
    X_train_final, y_train_final,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=64,
    callbacks=[early_stop],
    class_weight=class_weights
)

lstm_model.save(MODEL_SAVE_LSTM)

print("Training Complete!")


# ================= ACCURACY =================
print("\nLSTM Classification Report:\n")
lstm_probs = lstm_model.predict(X_test)
lstm_pred = np.argmax(lstm_probs, axis=1)
print(classification_report(y_test, lstm_pred, target_names=LABEL_NAMES))
lstm_acc = accuracy_score(y_test, lstm_pred)
print("\nLSTM Accuracy:", lstm_acc)


import matplotlib.pyplot as plt

plt.figure(figsize=(14,5))



# ================= ACCURACY CURVE =================
plt.subplot(1,2,1)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

# ================= LOSS CURVE =================
plt.subplot(1,2,2)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.show()



# ================= CONFUSION MATRIX =================
cm_lstm = confusion_matrix(y_test, lstm_pred)

plt.figure()
sns.heatmap(cm_lstm, annot=True, fmt='d',
            xticklabels=LABEL_NAMES,
            yticklabels=LABEL_NAMES)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("LSTM Confusion Matrix")
plt.show()



cm_lstm = confusion_matrix(y_test, lstm_pred)

plt.figure()
sns.heatmap(cm_lstm, annot=True, fmt='d',
            xticklabels=LABEL_NAMES,
            yticklabels=LABEL_NAMES)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("LSTM Confusion Matrix")
plt.show()



# ================= TEST FUNCTION =================
def test_on_file(csv_path):

    df = pd.read_csv(csv_path)

    if 'X_Acc' in df.columns:
        df.rename(columns={
            'X_Acc': 'acc_x', 'Y_Acc': 'acc_y', 'Z_Acc': 'acc_z',
            'X_Gyro': 'gyro_x', 'Y_Gyro': 'gyro_y', 'Z_Gyro': 'gyro_z'
        }, inplace=True)

    # Feature Engineering
    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    df['gyro_mag'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)

    
    data = df[FEATURES].values

    windows = [
        data[i:i + WINDOW_SIZE]
        for i in range(0, len(data) - WINDOW_SIZE, STEP_SIZE)
    ]

    if len(windows) == 0:
        print("Not enough data!")
        return

    X_val = np.array(windows)
    X_val = scaler.transform(X_val.reshape(-1, len(FEATURES))).reshape(len(X_val), WINDOW_SIZE, len(FEATURES))

   
    preds_lstm = lstm_model.predict(X_val, verbose=0) # verbose=0 hides the progress bar for cleaner output
    score_lstm = calculate_driving_score(preds_lstm)
    avg_probs_lstm = np.mean(preds_lstm, axis=0)
    pred_class_lstm = np.argmax(avg_probs_lstm)
    print("\n" + "="*40)
    print(" LSTM MODEL PREDICTION")
    print("="*40)
    print(f"Driving Score: {score_lstm}/100")
    print(f"Driving Style: {LABEL_NAMES[pred_class_lstm]}")
  


# ================= RUN TEST =================
print(f"Test File Name {TEST_FILE}")
if os.path.exists(TEST_FILE):
    test_on_file(TEST_FILE)
else:
    print("Test file not found!")
    
