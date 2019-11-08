from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Input,Model
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D,Dropout, MaxPooling2D, Flatten,BatchNormalization,GlobalAveragePooling2D,Activation,Add
#from sklearn.model_selection import StratifiedKFold,train_test_split
from tensorflow.keras import backend as K



class CNN_layer():
    def __init__(self):
        self.model = Sequential()
    def conv(self,input_dim, K=3, padding = "same", act = "relu",init = False, two_layer = False):
        
        if init:
            self.model.add(Conv2D(input_dim, kernel_size=(K,K),padding = "same" , input_shape=(32, 32, 3)))
        else:
            self.model.add(Conv2D(input_dim, kernel_size=(K,K),padding = "same"))
            
        if two_layer:
            self.model.add(BatchNormalization())
            self.model.add(Activation(act))
            self.model.add(Conv2D(input_dim, kernel_size=(K,K),padding = "same"))
            self.model.add(Activation(act))
        else:
            self.model.add(BatchNormalization())
            self.model.add(Activation(act))
       
    def pool(self,K):
        self.model.add(MaxPooling2D(pool_size=(K, K)))
        
    def self_add(self,method):
        self.model.add(method)
        
    def get_model(self,):
        return self.model
    
class Residual_CNN():
    def __init__(self,n_block,n_layer,init_fitler,K):
        self.input = Input(shape = (32,32,3))
        self.n_block = n_block
        self.n_layer = n_layer
        self.K = K
        self.init_filter = init_fitler
    def _residual_block(self, x, K = 3,n_filter=32):
        skip_x = x
        x = Conv2D(n_filter, K, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(n_filter, K, padding="same")(x)
        x = Add()([x, skip_x])
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        return x
    def _pool(self,x, K, max = True):
        return MaxPooling2D(pool_size=(K, K)) if max else MeanPooling2D(pool_size=(K, K))

    def build_model(self):
        x = self.input
        n_filter = self.init_filter

        for _ in range(self.n_block):
            x = Conv2D(n_filter, self.K, padding="same")(x)
            for _ in range(self.n_layer):
                x = self._residual_block(x,n_filter= n_filter)
            x = self._pool(x,2)
            n_filter *= 2

        x = GlobalAveragePooling2D()(x)
        y = Dense(10, activation="softmax")(x)
        model = Model(inputs=self.input, outputs=y)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
        return model

def pretrain_model():
    model =Xception(include_top = False, weights = "imagenet",input_shape = (71,71,3))
    x=model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(512,activation='relu')(x)
    x = Dropout(0.8)(x)
    x= Dense(256,activation='relu')(x)
    x = Dropout(0.8)(x)
    y=Dense(10,activation='softmax')(x)
    model = Model(inputs = model.input, outputs =y)

    for layer in model.layers[:30]:
        layer.trainable = False
        
        if layer.name.startswith('batch_normalization'):
            layer.trainable = True

    for layer in model.layers[30:]:
        layer.trainable = True
        #if layer.name.startswith('batch_normalization'):
        #    layer.trainable = True

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    return model

def pretrain_weight(model):

    base_model= VGG16(include_top = False, weights = "imagenet",input_shape =None)
    weights = [com.get_weights() for com in base_model.layers[1:]]

    model.layers[0].set_weights(weights[0])
    model.layers[3].set_weights(weights[1])
    model.layers[5].set_weights(weights[2])
    model.layers[6].set_weights(weights[3])
    #model.layers[9].set_weights(weights[4])
    return model
