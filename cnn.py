# -*- coding: utf-8 -*-
'''
This example demonstrates the demo for chemical-induced disease relation extraction.

'''

import pandas as pd
import numpy as np
import  cPickle
import matplotlib.pyplot as plt
from __future__ import print_function
from process_data import make_idx_data_cv
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.utils.vis_utils import plot_model
from utils import LossHistory,TwoWeightsHistory

def loadData(path,k=100):

    x = cPickle.load(open(path,"rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print(len(word_idx_map))
    print(len(vocab))
    print (len(revs))
    datasets = make_idx_data_cv(revs, word_idx_map, 1, max_l=max_l,k=k, filter_h=5)
    img_h = len(datasets[0][0])-1
    print ('img_h',img_h)
    print('max len',max_l)
    print(datasets[0].shape)
    test_set_x = datasets[1][:,:img_h]
    test_set_y = np.asarray(datasets[1][:,-1],"int32")
    train_set_x =datasets[0][:,:img_h]
    train_set_y =np.asarray(datasets[0][:,-1],"int32")
    print (np.shape(train_set_x))
    print('load data...')
    print(np.shape(W))
    print(type(W))
    return (train_set_x,train_set_y),(test_set_x,test_set_y),W


def runSigleCTDModel(savePath,func):
    np.random.seed(1337)

    batch_size = 32
    epoch =10
    print('Loading data...')

    # ***data path ***---

    dataPath ='corpus/wordseq/mr_newEntity.p'
    ctdPath ='corpus/ctd/transr/ctdData.p'
    # load data
    (X_train, y_train), (X_test, y_test), WordEm = loadData(path=dataPath, k=100)
    X_train2,X_test2,ctdEm = cPickle.load(open(ctdPath, "rb"))


    print('datapath:', dataPath)
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    train_label = to_categorical(y_train, 2)
    test_label = to_categorical(y_test, 2)

    print('set hyper-parameters:')
    max_features = (WordEm.shape[0])
    embedding_size = WordEm.shape[1]
    maxlen = X_train.shape[1]
    max_features2 =ctdEm.shape[0]
    embedding_size2 =ctdEm.shape[1]
    maxlen2 =X_train2.shape[1]

    print('Build model...')

    model = func(maxlen, max_features, embedding_size,
                        WordEm, maxlen2, max_features2, embedding_size2,
                        ctdEm)

    print('xxxxxx')
    pltname = savePath + '/modelcnn.png'
    import  os
    if os.path.isdir(savePath):
        pass
    else:
        os.makedirs(savePath)


    model.compile(

        loss='categorical_crossentropy',
                  optimizer='adagrad',
                  # optimizer='sgd',
                  metrics=['categorical_accuracy'])
    model.summary()
    # print (model.get_layer('patt_1').alfa)


    plot_model(model, to_file=pltname, show_shapes=True)
    print('Train...')
    # mean_label =np.zeros((X_train.shape[0],1))



    history = TwoWeightsHistory([X_test,X_test2])
    losshis = LossHistory()
    model.fit([X_train, X_train2], train_label, batch_size=batch_size,
              epochs=epoch, validation_split=0.2, shuffle=True,
              callbacks=[history, losshis])

    i = 0

    for result in history.result:
        i += 1
        np.savetxt(savePath+'/result_' + str(i) + '.txt', result, fmt="%.4f", delimiter=" ")

    print(losshis.epoch)
    plt.figure()
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    plt.sca(ax1)
    plt.plot(losshis.epoch, losshis.losses, '-or', label="loss")
    plt.plot(losshis.epoch, losshis.val_loss, '-xb', label="val_loss")
    # plt.xlim(0,2)
    plt.legend(loc='upper right')

    plt.sca(ax2)
    plt.plot(losshis.epoch, losshis.accuracy, '-or', label="accuracy")
    plt.plot(losshis.epoch, losshis.val_accuracy, '-xb', label="val_accuracy")
    plt.legend(loc='upper right')
    plt.savefig(savePath+'/Myfig.jpg')
    # plt.show()



if __name__=='__main__':


    from cdr_model import NAM_model

    from intra_evaluation import evaluate_for_corpus

    # experiment path

    save_path ='Trans_results/NAM'
    runSigleCTDModel(save_path, NAM_model)
    # evaluate_for_corpus(save_path)

