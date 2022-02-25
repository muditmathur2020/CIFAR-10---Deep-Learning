# Convolutional Neural Networks for CIFAR-10 


This repository is about some implementations of CNN Architecture  for **cifar10**.  

![cifar10][1]

I just use **Keras** and **Tensorflow** to implementate all of these CNN models.  
~~(maybe torch/pytorch version if I have time)~~  


## Requirements

- Python (3.5)
- keras (>= 2.1.5)
- tensorflow


## Model CNN - Sequential


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters= 32, kernel_size=(4,4), input_shape=(32,32,3), activation='relu'))
# POOLING LAYER
model.add(MaxPool2D(pool_size= (2,2)))

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters= 32, kernel_size=(4,4), input_shape=(32,32,3), activation='relu'))
# POOLING LAYER
model.add(MaxPool2D(pool_size= (2,2)))


# FLATTEN 
model.add(Flatten())

# Dense
model.add(Dense(256, activation='relu'))

# Dense [OUTPUT]
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics= ['accuracy'])
```



  [1]: ./images/cf10.png
