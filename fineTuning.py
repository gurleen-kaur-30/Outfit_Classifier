# from keras import backend
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import numpy as np

# path to the model weights files.
weights_path = 'weights/Full_weights.h5'
top_model_weights_path = 'saved_weights.h5'
# dimensions of our images.
img_width, img_height = 100, 200

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 264
nb_validation_samples = 60
epochs = 40
batch_size = 6

# build the VGG16 network

base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape = (img_width, img_height, 3))
print('Model loaded.')

print("The shape is....", base_model.output_shape[1:])

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(4, activation='sigmoid'))
 
# We now load the weights for the top model that have been obtained by 
# first training the top model (fineTuningInitial.py) and saving the weights.

top_model.load_weights(top_model_weights_path)


model = Model(inputs= base_model.input, outputs= top_model(base_model.output))

# We freeze the first 15 layers of the original architecture and set 
# then to non-trainable.
for layer in model.layers[:15]:
    layer.trainable = False

# we are compiling the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# we are preparing the data augmentation configuration.
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# fine-tune the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples//batch_size,
    verbose = 2)
validation_labels = np.array([0] * (20) + [1] * (12) + [2] * (8) + [3] * (20))
validation_labels = to_categorical(validation_labels)

print('acc : ', history.history['acc'] )
print('loss: ', history.history['loss'] )
