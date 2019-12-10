from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import h5py

# CNN design
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(26, activation='softmax'))
model.compile(optimizer=optimizers.SGD(lr=0.05), loss='categorical_crossentropy', metrics=['accuracy'])

# training and test data configuration
train_data = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_data = ImageDataGenerator(rescale=1. / 255)
training_data = train_data.flow_from_directory('data/train', target_size=(64, 64), batch_size=32,
                                               class_mode='categorical')
test_data = test_data.flow_from_directory('data/test', target_size=(64, 64), batch_size=32,
                                          class_mode='categorical')

# fitting the training data into model
final_model = model.fit_generator(training_data, steps_per_epoch=800, epochs=25,
                                  validation_data=test_data, validation_steps=6500)
# saving the weights
model.save('Weights.h5')

# list metrics collected
print(final_model.history.keys())

# Accuracy graph
plt.plot(final_model.history['accuracy'])
plt.plot(final_model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Loss Graph
plt.plot(final_model.history['loss'])
plt.plot(final_model.history['val_loss'])
plt.title('model loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
