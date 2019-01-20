#Data Preprocessing
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img, image

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
K.set_image_dim_ordering('th')

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 150, 150)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# the model so far outputs 3D feature maps (height, width, features)


model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


batch_size=16

#Augmentation Config
train_datagen=ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)


# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        '../TrainingData',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        '../TestingData',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=500 // batch_size,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=200 // batch_size)



#Display Data

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


img1 = image.load_img("../TrainingData/Empty/2013-01-17_10_40_06#006.jpg", target_size=(150, 150))
img2 = image.load_img("../TrainingData/Empty/2013-01-17_10_40_06#007.jpg", target_size=(150, 150))
img3 = image.load_img("../TrainingData/Occupied/2013-01-17_10_40_06#001.jpg", target_size=(150, 150))

images = [img1,img2,img3]

targets=[img1,img2,img3]

plt.figure(1)

for i in range(len(images)):

    targets[i] = np.expand_dims(img_to_array(targets[i]), axis=0)

    result = model.predict(targets[i])

    if result[0][0] >= 0.5:
        prediction = 'occupied'
    else:
        prediction = 'empty'

    print(prediction)
    print(result[0][0])
    ax1 = plt.subplot(1,3,i+1)
    plt.imshow(images[i])
    ax1.title.set_text(prediction)
    plt.axis('off')

plt.show()