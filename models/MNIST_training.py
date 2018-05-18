
# coding: utf-8

# In[1]:


from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import h5py


# In[13]:


batch_size = 64
num_classes = 10
epochs = 64


# In[3]:


# input image dimensions
img_rows, img_cols = 28, 28


# In[4]:


# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[5]:


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


# In[6]:


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# In[7]:


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# In[8]:


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[9]:


def mnist_model(input_shape, num_classes):
    
    X_input = Input(input_shape)
    
    X = Conv2D(32, kernel_size=(3, 3), activation='relu')(X_input)
    X = Conv2D(64, kernel_size=(3, 3), activation='relu')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = Dropout(0.25)(X)
    X = Flatten()(X)
    X = Dense(128, activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(num_classes, activation='softmax')(X)
    
    model = Model(X_input, X, name='MNIST model')
    
    return model


# In[10]:


model = mnist_model(input_shape, num_classes)


# In[11]:


model.compile(loss='categorical_crossentropy',optimizer='Adadelta', metrics=['accuracy'])


# In[14]:


model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))


# In[15]:


score = model.evaluate(x_test, y_test, verbose=0)


# In[16]:


print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[17]:


model.save('mnist_model.h5')

