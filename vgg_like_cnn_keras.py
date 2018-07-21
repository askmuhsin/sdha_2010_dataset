import numpy as np
import keras
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.optimizers import SGD
from get_data import load_data
from sklearn.model_selection import train_test_split

X, y = load_data()
X, y = np.asarray(X), np.asarray(y)
y = to_categorical(y)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    random_state=42)
print("Shape of training data : ", x_train.shape, y_train.shape)
print("Shape of testing data : ", x_test.shape, y_test.shape)


model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

print(model.summary())

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(x_train, y_train, batch_size=64, epochs=100)
score = model.evaluate(x_test, y_test, batch_size=32)
print("score :: ", score)

for _ in range(10):
    rand_img = random.randint(0, x_test.shape[0]-1)
    print("\n#########################")
    sample = x_test[rand_img].reshape(1, x_test[rand_img].shape[0], x_test[rand_img].shape[1], 1)
    print("\n******************")
    print(model.predict(sample))
    print(y_test[rand_img])
    print("\n******************")
    print("\n\n")

# import matplotlib.pyplot as plt
# plt.plot(x_test[1000], cmap='gray')
# plt.show()
