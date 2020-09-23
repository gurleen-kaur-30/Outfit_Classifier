import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# dimensions of our images.
img_width, img_height = 100, 200

top_model_weights_path = 'weights/Full_weights.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 264
nb_validation_samples = 60
epochs = 50
batch_size = 6


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open('bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)


def train_top_model():
    train_data = np.load('bottleneck_features_train.npy')
    train_labels = np.array( [0] * (81) + [1] * (66) + [2] * (44) + [3] * (73))
    train_labels = to_categorical(train_labels)


    validation_data = np.load('bottleneck_features_validation.npy')
    validation_labels = np.array([0] * (20) + [1] * (12) + [2] * (8) + [3] * (20))
    validation_labels = to_categorical(validation_labels)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    print (validation_data)
    print("the shape is................", validation_data.shape)
    print("the shape of labels is ............", validation_labels.shape)

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))


    model.save_weights("saved_weights.h5")

    (eval_loss, eval_accuracy) = model.evaluate( validation_data, validation_labels, batch_size=batch_size, verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))

    plt.figure(1)


# def predict():
#     # load the class_indices saved in the earlier step
#     class_dictionary = np.load('class_indices.npy').item()

#     num_classes = 4

#     # add the path to your test image below
#     image_path = ''

#     orig = cv2.imread(image_path)

#     print("[INFO] loading and preprocessing image...")
#     image = load_img(image_path, target_size=(224, 224))
#     image = img_to_array(image)

#     # important! otherwise the predictions will be '0'
#     image = image / 255

#     image = np.expand_dims(image, axis=0)

#     # build the VGG16 network
#     model = applications.VGG16(include_top=False, weights='imagenet')

#     # get the bottleneck prediction from the pre-trained VGG16 model
#     bottleneck_prediction = model.predict(image)

#     # build top model
#     model = Sequential()
#     model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
#     model.add(Dense(256, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(num_classes, activation='sigmoid'))

#     model.load_weights(top_model_weights_path)

#     # use the bottleneck prediction on the top model to get the final
#     # classification
#     class_predicted = model.predict_classes(bottleneck_prediction)

#     probabilities = model.predict_proba(bottleneck_prediction)

#     inID = class_predicted[0]

#     inv_map = {v: k for k, v in class_dictionary.items()}

#     label = inv_map[inID]

#     # get the prediction label
#     print("Image ID: {}, Label: {}".format(inID, label))

#     # display the predictions with the image
#     cv2.putText(orig, "Predicted: {}".format(label), (10, 30),
#                 cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2)

#     cv2.imshow("Classification", orig)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



save_bottlebeck_features()
train_top_model()