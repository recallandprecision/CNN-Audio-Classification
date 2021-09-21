import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt
#from keras.utils.vis_utils import plot_model

Xpickle = open("picklefiles/X.pickle","rb")
X = np.array(pickle.load(Xpickle))

Ypickle = open("picklefiles/y.pickle","rb")
y = np.array(pickle.load(Ypickle))




model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32, kernel_size=3, strides=3, padding="same", input_shape=X.shape[1:], activation='sigmoid'))
model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=3, padding="same", activation='sigmoid'))
model.add(tf.keras.layers.Conv2D(128, kernel_size=3, strides=3, padding="same", activation='sigmoid'))
model.add(tf.keras.layers.BatchNormalization(momentum=0.99))
model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=3, padding="same", activation='sigmoid'))
model.add(tf.keras.layers.Conv2D(32, kernel_size=3, strides=3, padding="same", activation='sigmoid'))


model.add(tf.keras.layers.Flatten())



model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
print("using tensorflow backend")
print("************************")
model.summary()





#For Bloack Diagram
'''from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
'''





model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(lr=.0001),metrics=['accuracy'])
#print(X)
#print(y)

history=model.fit(X, y, batch_size=3, epochs=30, validation_split=0.2)

    
model.save('cough_or_not.h5')

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train accuracy', 'test accuracy','train loss', 'test loss'], loc='upper left')
plt.show()
# summarize history for loss

'''plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()'''

