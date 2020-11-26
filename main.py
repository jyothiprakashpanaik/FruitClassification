# Import Lib
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator

print("[INFO] Creating CNN Model")
model = Sequential()
model.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=32,activation = 'relu'))
model.add(Dense(units=64,activation="relu"))
model.add(Dense(units=128,activation="relu"))
model.add(Dense(units=256,activation="relu"))
model.add(Dense(units=6,activation="softmax"))
model.summary()
model.compile(optimizer="adam",loss='categorical_crossentropy',metrics=["accuracy"])
print("[INFO] Compled the CNN Model")

print("[INFO] Training the CNN model")
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory("./Dataset/train",target_size=(64,64),batch_size=12,class_mode="categorical")
test_set = test_datagen.flow_from_directory("./Dataset/test",target_size=(64,64),batch_size=12,class_mode="categorical")

model.fit_generator(train_set,steps_per_epoch=101,epochs = 20,validation_data = test_set,validation_steps = 300)


model.save("model.h5")

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

print("[INFO] Model saved to disk")