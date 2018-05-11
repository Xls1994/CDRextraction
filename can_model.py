# -*- coding: utf-8 -*-
'''
author:yangyl

'''
from keras.models import Model
from keras.layers import  Dropout
from keras.layers import Embedding,Lambda,Input,Reshape,Flatten,RepeatVector,Add
from keras.layers import Conv1D, Dense,GlobalMaxPool1D,LSTM,AveragePooling1D
from keras.layers.merge import Concatenate,concatenate
from customize_layer import AttMemoryLayer,AttentivePoolingLayer
from keras.initializers import RandomUniform
from keras.regularizers import l2

def CAN(maxlen,max_features,embedding_size,WordEM,
                maxlen2,max_features2,embedding_size2,WordEM2):

    nb_filter = 100
    filter_sizes = (1,2,3,4)
    convs = []
    left_input = Input(shape=(maxlen,), dtype='int32', name='left_input')
    inputs = Embedding(max_features, embedding_size, input_length=maxlen,
                       weights=[WordEM],
                     )(left_input)
    right_input = Input(shape=(maxlen2,), dtype='int32', name='right_input')
    right = Embedding(max_features2, embedding_size2, input_length=maxlen2,trainable=True,
                      weights=[WordEM2],

                      )(right_input)
    for fsz in filter_sizes:
        conv = Conv1D(filters=nb_filter,
                      kernel_size=fsz,
                      padding='valid',
                      activation='relu',
                      )(inputs)

        # phrase-level attention
        pool =AttMemoryLayer(name='patt_'+str(fsz))([right,conv])
        relation =Flatten()(right)
        pool =Concatenate(1)([pool,relation])
        convs.append(pool)
    if len(filter_sizes) > 1:
        out = Concatenate(axis=1)(convs)
    else:
        out = convs[0]

   
    out =Dense(100,activation='relu')(out)
    out =Dropout(0.5)(out)
    out =Dense(50,activation='relu')(out)
    out =Dropout(0.5)(out)
    predict = Dense(2, activation='softmax')(out)

    model = Model(inputs=[left_input, right_input], outputs=predict)
    return model

def CAN_V2 (maxlen,max_features,embedding_size,WordEM,
                maxlen2,max_features2,embedding_size2,WordEM2):
    nb_filter = 100
    filter_sizes = (1,2,3,4)
    convs = []
    left_input = Input(shape=(maxlen,), dtype='int32', name='left_input')
    inputs = Embedding(max_features, embedding_size, input_length=maxlen,
                       weights=[WordEM],
                     )(left_input)
    right_input = Input(shape=(maxlen2,), dtype='int32', name='right_input')
    right = Embedding(max_features2, embedding_size2, input_length=maxlen2,trainable=True,
                      weights=[WordEM2],

                      )(right_input)
    for fsz in filter_sizes:
        conv = Conv1D(filters=nb_filter,
                      kernel_size=fsz,
                      padding='valid',
                      activation='relu',
                      )(inputs)

        # phrase-level attention
        relation = Flatten()(right)
        emb2 = RepeatVector(maxlen-fsz+1)(relation)
        conv = concatenate([conv, emb2], 2)
        pool = AttentivePoolingLayer(name="Attention")(conv)

        convs.append(pool)
    if len(filter_sizes) > 1:
        out = Concatenate(axis=1)(convs)
    else:
        out = convs[0]

   
    out =Dense(100,activation='relu')(out)
    out =Dropout(0.5)(out)
    out =Dense(50,activation='relu')(out)
    out =Dropout(0.5)(out)
    predict = Dense(2, activation='softmax')(out)

    model = Model(inputs=[left_input, right_input], outputs=predict)
    return model

if __name__ == '__main__':
    pass