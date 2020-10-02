from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils as ut
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.models import model_from_json
p=""
p=input("Enter the path where you extracted aimage folder ending with / = ")

json_file=open(p+"aimage/model.json",'r')
loaded_model_json=json_file.read();
json_file.close()
model=model_from_json(loaded_model_json)

model.compile(loss='mean_squared_error',
              optimizer='rmsprop',
              metrics=['accuracy'])

x=np.load(p+'aimage/data.npz')
label=LabelEncoder()
labels=label.fit_transform(np.array(x['y']))
b=ut.to_categorical(labels)
model.load_weights(p+"aimage/data_weights.h5")
dat=["eraser","pen","sharpner","pencil"]

i=0
r=0
for d in dat:
   dirs=os.listdir(p+'aimage/'+d)
   for file in dirs:
       n=plt.imread(p+'aimage/'+d+"/"+file)
       n = n.reshape((1,) + n.shape)
       print(n.shape)
       i+=1
       if d==dat[np.argmax(model.predict(n))]:
           r+=1
print(r/i)
