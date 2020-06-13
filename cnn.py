import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()
classifier.add(Conv2D(6, kernel_size=(5,5), activation="tanh", input_shape=(64, 64, 3)))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Conv2D(16, kernel_size=(5,5), activation="tanh"))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(120, activation="tanh"))
classifier.add(Dense(84, activation="tanh"))
classifier.add(Dense(5, activation="tanh"))
classifier.compile(loss = keras.metrics.mse, optimizer = keras.optimizers.Adam(), metrics=["accuracy"])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/home/murugesh/PycharmProjects/multi_class_classifier/assets',
                                                 target_size = (64, 64),
                                                 batch_size = 32)

test_set = test_datagen.flow_from_directory('/home/murugesh/PycharmProjects/multi_class_classifier/assets',
                                            target_size = (64, 64),
                                            batch_size = 32)

model = classifier.fit_generator(training_set,
                         steps_per_epoch = 80,
                         epochs = 10,
                         verbose=1,
                         validation_data = test_set,    
                         validation_steps = 2)

classifier.save("trained_weights.h5")
print("Saved model to disk")