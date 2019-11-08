import numpy as np
import matplotlib.pyplot as plt
import os, cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

RS = 1103

def load(hot = False):

    # 学習データ
    x_train = np.load("path")
    y_train = np.load("path")
    # テストデータ
    x_test = np.load("path")

    # this is case when want to transform category to onehot
    y_train_hot = np.eye(10)[y_train.astype('int32').flatten()]
    if hot:
        return x_train, x_test, y_train, y_train_hot
    return x_train, x_test, y_train 
def img_resize(data):
    #resize for inception net

    input_size = 71
    num=len(data)
    zeros = np.zeros((num,input_size,input_size,3))
    for i, img in enumerate(data):
        zeros[i] = cv2.resize(
            img,
            dsize = (input_size,input_size)
        )
    del data
    return zeros

def image_view(x_train):

    fig = plt.figure(figsize=(9, 15))
    fig.subplots_adjust(left=0, right=1, bottom=0,
                        top=0.5, hspace=0.05, wspace=0.05)

    for i in range(5):
        ax = fig.add_subplot(1, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(x_train[i])

def train_TA(all_train_x:np.array,train:list,val:list,batch_size:int):
    
    train_datagen = ImageDataGenerator( rotation_range = 10,
                                        channel_shift_range = 0.1,
                                        shear_range=2.0,
                                        zoom_range=0.2,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        horizontal_flip=True,
                                        #vertical_flip = True,
                                        # Global Contrast Normalization (GCN)
                                        samplewise_center=True,
                                        samplewise_std_normalization=True,
                                        zca_whitening=False)
    valid_datagen = ImageDataGenerator( # Global Contrast Normalization (GCN)
                                    samplewise_center=True,
                                    samplewise_std_normalization=True,)

    train_datagen.fit(all_train_x)
    train_generator = train_datagen.flow(train[0], train[1], batch_size=batch_size)
    
    
    valid_datagen = ImageDataGenerator( # Global Contrast Normalization (GCN)
                                    samplewise_center=True,
                                    samplewise_std_normalization=True,)
    valid_datagen.fit(val[0])
    valid_generator = valid_datagen.flow(val[0], val[1], batch_size=batch_size shuffle=False)
    return train_generator,valid_generator

def test_TA(test:np.array,batch_size:int):
    test_datagen = ImageDataGenerator( rotation_range = 10,
                                        channel_shift_range = 0.1,
                                        shear_range=2.0,
                                        zoom_range=0.2,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        horizontal_flip=True,
                                        #vertical_flip = True,
                                        # Global Contrast Normalization (GCN)
                                        samplewise_center=True,
                                        samplewise_std_normalization=True,
                                        zca_whitening=False)
    test_datagen.fit(test)
    test_generator = test_datagen.flow(test, batch_size=batch_size)
    return test_generator

def get_argment_generator(train_split,val_split,all_train,test,BATCH_SIZE):
    tr_gen,val_gen = train_TA(all_train,train_split,val_split,BATCH_SIZE)
    test_gen = test_TA(test,BATCH_SIZE)
    dummy_test_gen = test_TA(val_split[0],BATCH_SIZE) 
    return tr_gen,val_gen,dummy_test_gen,test_gen

def data_split(x_train,y_train, n_val):
    tr_x,val_x,tr_y,val_y = train_test_split(x_train,y_train,test_size = n_val,random_state = RS)
    return [tr_x,tr_y], [val_x,val_y]