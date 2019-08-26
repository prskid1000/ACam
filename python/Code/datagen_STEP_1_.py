# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import os

p=""
p=input("Enter the path where you extracted aimage folder ending with / = ")

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')
        
dat=["eraser","pen","sharpner","pencil"]
dirs=os.listdir(p+'aimage/source')

for x in dat:
    os.mkdir(p+'aimage/'+x)
    
for y in dat:
  for files in dirs:
      if files.startswith(y):
          z=p+'aimage/source/'+y+".jpg"
          img = (z)
          x = plt.imread(img)
          x = x.reshape((1,) + x.shape)
          i = 0
          for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=p+'aimage'+"/"+y,save_prefix=y, save_format='png'):
                          i += 1
                          print(i)
                          if i >10000:
                              break;
