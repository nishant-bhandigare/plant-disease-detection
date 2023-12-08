import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#15 types of plant disease cases are present in the datase, so 15 cases can be predicted
num_classes = 15

train_data_gen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)

#creating the traing data
train_data = train_data_gen.flow_from_directory(
    r"Plant Disease Prediction\Dataset\PlantVillage",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training')


#creating the testing data
val_data = train_data_gen.flow_from_directory(
    r"Plant Disease Prediction\Dataset\PlantVillage",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation')

#printing the assigned indices to each class
print(train_data.class_indices)
print(val_data.class_indices)

#creating the model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

#compiling the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#training the model
model.fit(train_data,
        epochs=10,
        validation_data=val_data)

# model.fit_generator(train_data, epochs=10, validation_data=val_dat)

#saving the model for future use
model.save_weights("plant_disease_detection_weights.h5")
model.save('plant_disease_detection.h5')
print("Model saved to disk")
    
test_data = train_data_gen.flow_from_directory(
    r"Plant Disease Prediction\Dataset\PlantVillage_Test",
    target_size=(224, 224), 
    # batch_size=32,
    batch_size=23,
    class_mode='categorical')

print(test_data.class_indices)

test_loss, test_acc = model.evaluate(test_data)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)