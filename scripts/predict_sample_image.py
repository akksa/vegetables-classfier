import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model('/Users/saravana/AI/Lottery/scripts/veg.h5')
path = '/Users/saravana/AI/datasets/VegImages/TRAINING/Tomato/IMG_20200522_092148.jpg'
img = image.load_img(path, target_size=(300, 300))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes)
print(classes[0])
