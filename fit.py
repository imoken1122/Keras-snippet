import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold,KFold
from model import Residual_CNN
from sklearn.metrics import accuracy_score
import os
from data_utils import train_TA,test_TA,data_split,load,get_argment_generator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
N_VAL = 5000
BATCH_SIZE = 256
N_BLOCK = 100
N_LAYER = 3
FILTER = 32
KERNEL = 3
EPOCH = 100
N_ENSMBLE = 10
PATH = ""

def TTA(model,generator,sub_epoch):
    tta_pred = 0
    for e in range(sub_epoch):
        tta_pred += model.predit(generator)
    return tta_pred/sub_epoch

def TTA_fitting(tr_gen,val_gen,test_gen,dummy_test_gen):
    fold_val,fold_test = [],[]
    for i in range(N_ENSMBLE):
        callbacks = [
        # val_lossが下がらなくなった際に学習を終了するコールバック
        EarlyStopping(monitor='val_loss', patience=30, verbose=1),
        # val_lossが下がらなくなった際に学習率を下げるコールバック
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-9),
        # val_categorical_accuracyが最大のmodelを保存するコールバック
        ModelCheckpoint(savePath, monitor='val_categorical_accuracy', save_best_only=True)
        ]    
        
        # load data
        RCNN = Residual_CNN(N_BLOCK,N_LAYER,FILTER,KERNEL)
        model = RCNN.build_model()

        # model training
        model.fit_generator(tr_gen, callbacks,steps_per_epoch=tr_x.shape[0] // BATCH_SIZE,
                        epochs=EPOCH, validation_data=val_gen,validation_steps=val_x.shape[0] // BATCH_SIZE)
        # TTA dummy test data
        val_pred = TTA(model, dummy_test_gen,sub_epoch=10)
        tta_val_acc = accuracy_score(np.argmax(val_y, 1), np.argmax(val_pred, 1))
        print(f"TTA_vaild_accuracy : {tta_val_acc}")

        # TTA test data
        test_pred = TTA(model,test_gen,sub_epoch = 10)

        # stacking TTA prediction
        fold_val.append(val_pred)
        fold_test.append(test_pred)

        # TTA ensmble accuracy
        if i!=0:
            mean_val_pred = np.mean(fold_val,axis = 0)
            mean_test_pred = np.mean(fold_test,axis = 0)
            ensmble_tta_val_acc = accuracy_score(np.argmax(val_y, 1), np.argmax(mean_val_pred, 1))
            print(f"ensmbleTTA_vaild_accuracy : {ensmble_tta_val_acc}") 

            # saving TTA ensmble prediction of test data
            submission = pd.Series(mean_test_pred, name='label')
            submission.to_csv(os.path.join(PATH, 'submission_ensemble_' + str(i + 1) + ".csv"), header=True, index_label='id')

def Fold_fitting(train_x,train_y,test):
    N_FOLD = 5
    RCNN = Residual_CNN(N_BLOCK,N_LAYER,FILTER,KERNEL)
    model = RCNN.build_model()

    fold = StratifiedKFold(n_splits=N_FOLD,random_state = 1103)
    pred = []
    acc = []
    for i,(tr_idx,val_idx) in enumerate(fold.split(train_x,np.argmax(train_y,1))):
        callbacks = [
                # val_lossが下がらなくなった際に学習を終了するコールバック
                EarlyStopping(monitor='val_loss', patience=30, verbose=1),
                # val_lossが下がらなくなった際に学習率を下げるコールバック
                ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-9),
                # val_categorical_accuracyが最大のmodelを保存するコールバック
                ModelCheckpoint(savePath, monitor='val_categorical_accuracy', save_best_only=True)
                ]    
        
        print(f"==========={i}_fold start ================= ")
        train_split= train_x[tr_idx],train_y[val_idx]
        val_split = val_y[tr_idx],val_y[val_idx]
        tr_gen,val_gen,test_gen,dummy_test_gen = get_argment_generator(train_split,val_split,train,test,BATCH_SIZE)

        h = model.fit_generator(tr_gen, epochs=EPOCH, steps_per_epoch = len(tr_x)//BATCH_SIZE,
                                validation_data = val_gen, validation_steps =len(val_x)//BATCH_SIZE,
                                callbacks=callbacks)
        val_pred = model.predict(dummy_test_gen)
        test_pred = model.predict(test_gen)
        val_score = accuracy_score(np.argmax(val_split[1], 1),val_pred)
        print(f"fold_{i} val accuracy : {val_score}")
        pred.append(test_pred)

    mean_pred = np.mean(pred,axis=0)
    pred_y = np.argmax(mean_pred, 1)
    submission = pd.Series(pred_y, name='label')
    submission.to_csv(os.path.join(PATH, f"submission_fold{N_FOLD}.csv"), header=True, index_label='id')


train,test,y = load()
train_split,val_split = data_split(train,y,N_VAL)
tr_gen,val_gen,test_gen,dummy_test_gen = get_argment_generator(train_split,val_split,train,test,BATCH_SIZE)
tr_x,tr_y = train_split
val_x,val_y = val_split 
