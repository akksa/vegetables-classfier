import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
    
print('Loaded Libs')

TRAIN_DIR = '/Users/saravana/AI/datasets/VegImages/TRAINING'
VALIDATION_DIR = '/Users/saravana/AI/datasets/VegImages/VALIDATION'


training_datagen = ImageDataGenerator(rescale=1./255,
                                      rotation_range=40,
                                      width_shift_range=0.2,
                                      height_shift_range = 0.2,
                                      shear_range = 0.2,
                                      zoom_range = 0.2,
                                      horizontal_flip=True,
                                      fill_mode = 'nearest'
                                     )
validation_datagen = ImageDataGenerator(rescale=1./255)

train_genrator =training_datagen.flow_from_directory(TRAIN_DIR,target_size=(300,300),class_mode='categorical',batch_size=64)

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, target_size=(300,300),class_mode='categorical',batch_size=64)

print('Traininig and Validation Image generated')
print('Started CNN')
model = tf.keras.Sequential([
    
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(300,300,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
    
])

print(model.summary())
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
history = model.fit(train_genrator, epochs=25, validation_data= validation_generator, verbose=1, steps_per_epoch=4, validation_steps=2)
model.save('veg.h5')
print('Done')