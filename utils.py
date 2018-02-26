# -*- coding:utf-8 -*-
'''
author: yangyl

'''

import keras.callbacks as callbacks

# attention_mat =K.function([model.layers[0].input],[model.get_layer('att').alfa])
class TwoWeightsHistory(callbacks.Callback):
    def __init__(self,data):
        self.data=data
        super(TwoWeightsHistory,self).__init__()
    def on_train_begin(self, logs={}):
        self.result = []

    def on_epoch_end(self, epoch, logs=None):
        X_test, X_test2 =self.data
        result = self.model.predict([X_test, X_test2])
        print ('\n')
        # print ('epoch.{}'.format(epoch),attention_mat([X_test,X_test2]))
        # attentionMAT=attention_mat([X_test])
        # attentionMAT=np.squeeze(attentionMAT)
        # print (np.shape(attentionMAT))
        # np.savetxt(savePath+'/attmat{}'.format(epoch)+'.txt',attentionMAT
        #            ,fmt="%.4f",delimiter=' '
        #     )
        self.result.append(result)

class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_loss = []
        self.epoch = []
        self.accuracy = []
        self.val_accuracy = []

    def on_epoch_end(self, epoch, logs=None):
        self.epoch.append(epoch)
        self.losses.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.accuracy.append(logs.get('categorical_accuracy'))
        self.val_accuracy.append(logs.get('val_categorical_accuracy'))
