# -------------------------
# 1. Import Libraries
# -------------------------
import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import itertools

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Dense, Dropout
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image

import warnings
warnings.filterwarnings('ignore')

# -------------------------
# 2. Load Dataset
# -------------------------
file_path = 'dataset'

# List all classes
name_class = os.listdir(file_path)
print("Classes:", name_class)

# Get all filepaths
filepaths = list(glob.glob(file_path + '/**/*.*'))
print(f"Total images found: {len(filepaths)}")

# Extract labels
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

# Create DataFrame
filepath_series = pd.Series(filepaths, name='Filepath').astype(str)
labels_series = pd.Series(labels, name='Label')
data = pd.concat([filepath_series, labels_series], axis=1)
data = data.sample(frac=1).reset_index(drop=True)  # shuffle
print(data.head())

# -------------------------
# 3. EDA (Exploratory Data Analysis)
# -------------------------
# Class distribution
plt.figure(figsize=(8,5))
sns.countplot(x='Label', data=data, order=data['Label'].value_counts().index)
plt.title('Number of Images per Class')
plt.xticks(rotation=45)
plt.show()

# Check image dimensions
sample_img = plt.imread(data['Filepath'][0])
print(f"Sample image shape: {sample_img.shape}")

# Visualize few images
fig, axes = plt.subplots(2, 4, figsize=(12,6))
for ax, (img_path, label) in zip(axes.flatten(), zip(data['Filepath'][:8], data['Label'][:8])):
    img = plt.imread(img_path)
    ax.imshow(img)
    ax.set_title(label)
    ax.axis('off')
plt.tight_layout()
plt.show()

# -------------------------
# 4. Train-Test Split
# -------------------------
train, test = train_test_split(data, test_size=0.2, random_state=42)
print(f"Training samples: {len(train)}, Testing samples: {len(test)}")

# -------------------------
# 5. Data Preprocessing and Augmentation
# -------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train,
    x_col='Filepath',
    y_col='Label',
    target_size=(256, 256),
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42
)

valid_gen = test_datagen.flow_from_dataframe(
    dataframe=test,
    x_col='Filepath',
    y_col='Label',
    target_size=(256, 256),
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

# -------------------------
# 6. Model Building (Transfer Learning with ResNet50)
# -------------------------
pretrained_model = ResNet50(
    input_shape=(256, 256, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

pretrained_model.trainable = False

x = Dense(128, activation="relu")(pretrained_model.output)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
outputs = Dense(8, activation='softmax')(x)

model = Model(inputs=pretrained_model.input, outputs=outputs)

model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

# -------------------------
# 7. Model Training
# -------------------------
history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=20,
)

# -------------------------
# 8. Training Curves
# -------------------------
pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()
plt.title('Training vs Validation Accuracy')
plt.show()

pd.DataFrame(history.history)[['loss', 'val_loss']].plot()
plt.title('Training vs Validation Loss')
plt.show()

# -------------------------
# 9. Evaluation and Confusion Matrix
# -------------------------
# Evaluate
results = model.evaluate(valid_gen, verbose=0)
print(f"Test Loss: {results[0]:.5f}")
print(f"Test Accuracy: {results[1]*100:.2f}%")

# Predictions
predictions = model.predict(valid_gen)
y_pred = np.argmax(predictions, axis=1)

# True labels
y_true = valid_gen.classes

# Labels Mapping
labels_map = train_gen.class_indices
labels_map = dict((v,k) for k,v in labels_map.items())

# Classification report
print(classification_report(y_true, y_pred, target_names=list(labels_map.values())))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(labels_map.values()),
            yticklabels=list(labels_map.values()))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# -------------------------
# 10. Save the Model
# -------------------------
model.save("model_blood_group_detection_resnet.h5")
print("Model saved successfully!")

# -------------------------
# 11. Single Image Prediction Example
# -------------------------
# Load model
model = load_model('model_blood_group_detection_resnet.h5')

# Single image prediction
img_path = 'C:\Users\ADMIN\Documents\SEM-6\DL PROJECT\dataset\AB+\augmented_cluster_4_4.BMP'  # Update this path as needed
img = image.load_img(img_path, target_size=(256, 256))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
predicted_class = np.argmax(preds)
confidence = preds[0][predicted_class] * 100

# Get label
predicted_label = labels_map[predicted_class]

# Show image
plt.imshow(image.array_to_img(img))
plt.axis('off')
plt.title(f"Prediction: {predicted_label} ({confidence:.2f}%)")
plt.show()
