# ------------------------------------------------------
# 1. Import Libraries
# ------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ------------------------------------------------------
# 2. Load Dataset
# ------------------------------------------------------

data_dir = 'dataset'  # <-- Replace with your dataset folder
categories = os.listdir(data_dir)

data = []
for category in categories:
    category_path = os.path.join(data_dir, category)
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        data.append((img_path, category))

data = pd.DataFrame(data, columns=['Filepath', 'Label'])

print(f"Total samples: {len(data)}")
print(data.head())

# ------------------------------------------------------
# 3. Exploratory Data Analysis (EDA)
# ------------------------------------------------------

# Class distribution
plt.figure(figsize=(8,6))
sns.countplot(x='Label', data=data)
plt.title('Blood Group Class Distribution')
plt.xticks(rotation=45)
plt.show()

# Display few images
plt.figure(figsize=(12,8))
for i in range(9):
    sample = data.sample(n=1).iloc[0]
    img = cv2.imread(sample['Filepath'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(3,3,i+1)
    plt.imshow(img)
    plt.title(sample['Label'])
    plt.axis('off')
plt.tight_layout()
plt.show()

# ------------------------------------------------------
# 4. Train-Validation-Test Split
# ------------------------------------------------------

train, temp = train_test_split(data, test_size=0.3, random_state=42, stratify=data['Label'])
valid, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['Label'])

print(f"Training samples: {len(train)}")
print(f"Validation samples: {len(valid)}")
print(f"Testing samples: {len(test)}")

# ------------------------------------------------------
# 5. Preprocessing (Image Augmentation + Scaling)
# ------------------------------------------------------

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

target_size = (224, 224)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train,
    x_col='Filepath',
    y_col='Label',
    target_size=target_size,
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42
)

valid_gen = valid_datagen.flow_from_dataframe(
    dataframe=valid,
    x_col='Filepath',
    y_col='Label',
    target_size=target_size,
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

# ------------------------------------------------------
# 6. Load MobileNetV2 Base Model
# ------------------------------------------------------

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))

# ------------------------------------------------------
# 7. Freeze Base Layers
# ------------------------------------------------------

for layer in base_model.layers:
    layer.trainable = False

# ------------------------------------------------------
# 8. Define Function to Build Model (for tuning)
# ------------------------------------------------------

def build_model(dropout_rate=0.3, learning_rate=0.001):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(len(categories), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# ------------------------------------------------------
# 9. Hyperparameter Tuning (Manual)
# ------------------------------------------------------

dropout_rates = [0.3, 0.4]
learning_rates = [0.001, 0.0005]

best_val_accuracy = 0
best_model = None
best_params = {}

for dr in dropout_rates:
    for lr in learning_rates:
        print(f"\nTraining with Dropout: {dr}, Learning Rate: {lr}")

        model = build_model(dropout_rate=dr, learning_rate=lr)

        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
            ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
        ]

        history = model.fit(
            train_gen,
            validation_data=valid_gen,
            epochs=20,
            callbacks=callbacks,
            verbose=1
        )

        val_acc = max(history.history['val_accuracy'])
        print(f"Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model = model
            best_params = {'dropout_rate': dr, 'learning_rate': lr}

print("\nBest Parameters:", best_params)
print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")

# ------------------------------------------------------
# 10. Fine-tuning (Unfreeze some base layers)
# ------------------------------------------------------

# Unfreeze last 30 layers for fine-tuning
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Recompile
optimizer = tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate'] / 10)
best_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train again
history_finetune = best_model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=10,
    callbacks=[
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ],
    verbose=1
)

# ------------------------------------------------------
# 11. Save the Final Model
# ------------------------------------------------------

best_model.save('bloodgroup_mobilenet_finetuned.h5')
print("Fine-tuned model saved as bloodgroup_mobilenet_finetuned.h5")

# ------------------------------------------------------
# 12. Evaluate the Final Model
# ------------------------------------------------------

# Accuracy and Loss plots
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.plot(history_finetune.history['accuracy'], label='Fine-tuned Train Accuracy')
plt.plot(history_finetune.history['val_accuracy'], label='Fine-tuned Validation Accuracy')
plt.title('Fine-tuned Model Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history_finetune.history['loss'], label='Fine-tuned Train Loss')
plt.plot(history_finetune.history['val_loss'], label='Fine-tuned Validation Loss')
plt.title('Fine-tuned Model Loss')
plt.legend()

plt.show()

# ------------------------------------------------------
# 13. Prediction on Single Image (User Input)
# ------------------------------------------------------

# Load the fine-tuned model
model = load_model('bloodgroup_mobilenet_finetuned.h5')

# Define the class labels
labels = {'A+': 0, 'A-': 1, 'AB+': 2, 'AB-': 3, 'B+': 4, 'B-': 5, 'O+': 6, 'O-': 7}
labels = dict((v, k) for k, v in labels.items())

# Example: Single image prediction
img_path = 'dataset/AB+/augmented_cluster_4_4.BMP'

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

result = model.predict(x)
predicted_class = np.argmax(result)
predicted_label = labels[predicted_class]
confidence = result[0][predicted_class] * 100

plt.imshow(image.array_to_img(image.img_to_array(img)/255.0))
plt.axis('off')
plt.title(f"Prediction: {predicted_label} with confidence {confidence:.2f}%")
plt.show()
