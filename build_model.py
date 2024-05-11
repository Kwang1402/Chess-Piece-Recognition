from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from glob import glob

img_width = 256
img_height = 256
batch_size = 32
num_of_epochs = 100


num_of_classes = 6

# data augmentation

TRAIN_DIR = "train"

train_datagen = ImageDataGenerator(rescale=1/255.0,
                                   rotation_range=30,
                                   zoom_range=0.4,
                                   horizontal_flip=True,
                                   shear_range=0.4)

train_gen = train_datagen.flow_from_directory(TRAIN_DIR,
                                              batch_size=batch_size,
                                              class_mode='categorical',
                                              target_size=(img_height, img_width))

VALIDATION_DIR = "validation"

val_datagen = ImageDataGenerator(rescale = 1/255.0)

val_gen = val_datagen.flow_from_directory(VALIDATION_DIR,
                                          batch_size=batch_size,
                                          class_mode='categorical',
                                          target_size=(img_height, img_width))

# early stopping

call_back = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

# save the best model

best_model_file = "chess_best_model.keras"

best_model = ModelCheckpoint(best_model_file, monitor='val_accuracy', verbose=1, save_best_only=True)

# build model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),

    Dense(512, activation='relu'),
    Dense(512, activation='relu'),

    Dense(num_of_classes, activation='softmax')
])

print(model.summary())

# compile model with Adam optimizer
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_gen,
                    epochs=num_of_epochs,
                    verbose=1,
                    validation_data=val_gen,
                    callbacks=[best_model]
                    )

# plot the result
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

# accuracy chart

fig = plt.figure(figsize=(14,7))
plt.plot(epochs, acc , 'r', label="Train accuracy")
plt.plot(epochs, val_acc , 'b', label="Validation accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train and validation accuracy')
plt.legend(loc='lower right')
plt.show()

#loss chart
fig2 = plt.figure(figsize=(14,7))
plt.plot(epochs, loss , 'r', label="Train loss")
plt.plot(epochs, val_loss , 'b', label="Validation loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and validation Loss')
plt.legend(loc='upper right')
plt.show()