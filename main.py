import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tenserflow as tf

mnist=tf.keras.datassets.mnist
(x_traain, y_train) ,(x_test,y_test)=mnist.load_data()

x_train=tf.keras.utlis.normalize(x_train,axis-1)
x_test=tf.keras.utlis.normalize(x_train,axis-1)

model_.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))

model.add(tf.keras.layers.Dense(units=128,activation=tf.mn.relu))
model.add(tf.keras.layers.Dense(units=128,activation=tf.mn.relu))
model.add(tf.keras.layers.Dense(units=10,activation=tf.mn.softmax))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train, epochs=3)

accuracy,loss=model.evaluate(x_test,y_test)
print(accuracy)
print(loss)

model.save('digits.model')

for x in range(2,8):
    img=cv.imread(f'(x).png')[:,:,0]
img=np.invert(np.array([img]))
prediction=model.predict(img)
print(np.argmax(prediction))
plt.inshow(img[0],cmap=plt.cm.binary)
plt.show()