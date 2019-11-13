from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Conv1D, CuDNNLSTM, Activation, SimpleRNN,LSTM,GRU,Bidirectional,Dropout,Input,Add,Permute,Reshape,Multiply,Flatten,Lambda,RepeatVector,Dot,concatenate,Concatenate,BatchNormalization,MaxPooling1D,GlobalMaxPooling1D
import pandas as pd

class Att_LSTM():
    def __init__(self,n_feature,n_class):
        self.n_feature = n_feature
        self.n_class = n_class
    def Attention(self,x):
        tanh_H = Activation('tanh')(x)
        a = Dense(1,activation="softmax")(tanh_H)
        r = Dot(axes=1)([x,a])
        r = Activation('tanh')(r)
        r = Flatten()(r)
        return r
    def build_model(self):
        hid_dim = 64
        ts =n_feature
        input1 =Input((ts//2,1))
        input2 = Input((ts//2,1))

        x1 =Bidirectional(CuDNNLSTM(hid_dim,return_sequences=True))(input1)
        x1 = Dropout(0.5)(x1)
        x1 =Bidirectional(CuDNNLSTM(hid_dim,return_sequences=True))(x1)
        #x1 =CuDNNLSTM(hid_dim,return_sequences=True)(input1)
        #x1 = Attention(x1)
        x1 = Model(inputs = input1, outputs = x1)
        
        x2 =Bidirectional(CuDNNLSTM(hid_dim,return_sequences=True))(input2)
        x2 =Dropout(0.5)(x2)
        x2 =Bidirectional(CuDNNLSTM(hid_dim,return_sequences=True))(input2)
        #x2 =CuDNNLSTM(hid_dim,return_sequences=True)(input2)
        #x2 = Attention(x2)
        x2 = Model(inputs = input2, outputs = x2)
        
        comb = concatenate([x1.output,x2.output])
        x =Bidirectional(CuDNNLSTM(hid_dim,return_sequences=True),merge_mode ="sum")(comb)
        x = self.Attention(x)
        x = Dense(32, activation = "relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(self.n_class,activation = "softmax")(x)
        return Model(inputs = [x1.input,x2.input], outputs = x)