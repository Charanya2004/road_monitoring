import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation for training images
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.15, 
                                   width_shift_range=0.2, height_shift_range=0.2, 
                                   shear_range=0.15, horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    'path_to_training_data',  # Replace with your training data path
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')

# Building a CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax')  # Adjust number of classes based on your dataset
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10, steps_per_epoch=100)

# Save the model
model.save('road_condition_model.h5')
