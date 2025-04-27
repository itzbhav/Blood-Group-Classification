# ------------------------------------------------------
# 1. Import Libraries
# ------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input

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
# 7. Freeze Layers
# ------------------------------------------------------

for layer in base_model.layers:
    layer.trainable = False

# ------------------------------------------------------
# 8. Add Custom Layers
# ------------------------------------------------------

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(len(categories), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ------------------------------------------------------
# 9. Compile and Train the Model
# ------------------------------------------------------

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=20
)

# ------------------------------------------------------
# 10. Save the Model
# ------------------------------------------------------

model.save('bloodgroup_mobilenet_model.h5')
print("Model saved as bloodgroup_mobilenet_model.h5")

# ------------------------------------------------------
# 11. Evaluate the Model
# ------------------------------------------------------

# Accuracy and Loss plots
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.legend()

plt.show()

# ------------------------------------------------------
# 12. Prediction on Single Image (User Input)
# ------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

# Load the pre-t rained model
model = load_model('bloodgroup_mobilenet_model.h5')

# Define the class labels
labels = {'A+': 0, 'A-': 1, 'AB+': 2, 'AB-': 3, 'B+': 4, 'B-': 5, 'O+': 6, 'O-': 7}
labels = dict((v, k) for k, v in labels.items())

# Example of loading a single image and making a prediction
img_path = 'dataset/AB+/augmented_cluster_4_4.BMP'

# Preprocess the image accordingly (check the model's expected input dimensions)
img = image.load_img(img_path, target_size=(224, 224))  # Example target size for AlexNet (224x224)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)  # Ensure this matches the model's preprocessing function

# Make prediction
result = model.predict(x)
predicted_class = np.argmax(result)  # Get the predicted class index

# Map the predicted class to the label
predicted_label = labels[predicted_class]
confidence = result[0][predicted_class] * 100  # Confidence level

# Display the image
plt.imshow(image.array_to_img(image.img_to_array(img) / 255.0))
plt.axis('off')  # Hide axes

# Display the prediction and confidence below the image
plt.title(f"Prediction: {predicted_label} with confidence {confidence:.2f}%")
plt.show()
