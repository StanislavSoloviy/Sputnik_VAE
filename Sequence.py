from skimage.io import imread
from skimage.transform import resize
import numpy as np
import math
import tensorflow as tf
import main
import cv2
from keras import layers

# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

class DataSequence(tf.keras.utils.Sequence):

    def __init__(self, x_set, y_set=None, batch_size=None):
        self.x = x_set
        self.y = x_set
        self.batch_size = main.BATCH_SIZE
        self.hight = main.IMG_SHAPE
        self.weith = main.IMG_SHAPE


    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        low = idx * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        high = min(low + self.batch_size, len(self.x))
        batch_x = self.x[low:high]
        temp = np.array([self.processing(file_name) for file_name in batch_x])
        temp = temp.reshape((self.batch_size, self.hight, self.weith, 3)) / 255.
        return np.where(temp > .5, 1.0, 0.0).astype('float32')


    def processing(self, file):
        """Преобразование изображения в вектор."""
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.hight, self.weith))
        # img = np.array(img).reshape(self.hight, self.weith, 3)
        # img = img / 255.0
        img = self.augmentataion(img)
        return img

    def augmentataion(self, image):
        image = layers.RandomFlip("horizontal")(image)
        image = layers.RandomRotation(0.1)(image)
        image = layers.RandomZoom(0.1)(image)
        return image
