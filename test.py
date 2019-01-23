from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img, image

# datagen = ImageDataGenerator(
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     rescale=1./255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
K.set_image_dim_ordering('th')

# model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=(3, 150, 150)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))


# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))


# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))


# model.add(Flatten()) 
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))

# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])


# batch_size=16

# train_datagen=ImageDataGenerator(
#     rescale=1./255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )

# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#         '../TrainingData',  
#         target_size=(150, 150),  
#         batch_size=batch_size,
#         class_mode='binary')  

# validation_generator = test_datagen.flow_from_directory(
#         '../TestingData',
#         target_size=(150, 150),
#         batch_size=batch_size,
#         class_mode='binary')

# model.fit_generator(
#         train_generator,
#         steps_per_epoch=500 // batch_size,
#         epochs=10,
#         validation_data= validation_generator,
#         validation_steps=200 // batch_size)

# model.save('my_model.h5') # Allows model loading with new_model = keras.models.load_model('my_model.h5') // new_model.summary()

model = load_model('my_model.h5')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib.patches as patches

import glob
images = []
for filename in glob.glob('../TestingData/*.jpg'):
    im=image.load_img(filename, target_size=(150,150))
    images.append(im)

targets = []

plt.figure(1)

for i in range(len(images)):
    targets.append(images[i])
    targets[i] = np.expand_dims(img_to_array(targets[i]), axis=0)
    result = model.predict(targets[i])
    c = ''
    if result[0][0] >= 0.5:
        prediction = 'occupied'
        c = 'r'
    else:
        prediction = 'empty'
        c = 'g'

    ax1 = plt.subplot(6,5,i+1)
    plt.imshow(images[i])
    # Create a Rectangle patch
    rect = patches.Rectangle((0,0),images[i].width,images[i].height,linewidth=1,edgecolor=c,facecolor='none')
    # Add the patch to the Axes
    ax1.add_patch(rect)
    ax1.title.set_text(prediction + ": " + str(result[0][0] * 100) + "%")
    plt.axis('off')
    plt.subplots_adjust(hspace = 0.6)
plt.show()
