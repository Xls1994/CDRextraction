# -*- coding: utf-8 -*-
'''
author:yangyl

'''
from keras.models import Model
from keras.layers import  Dropout
from keras.layers import Embedding,Input,Flatten,RepeatVector,Reshape
from keras.layers import Conv1D, Dense,GlobalMaxPool1D,LSTM
from keras.layers.merge import Concatenate,concatenate
from customize_layer import AttMemoryLayer,  AttentivePoolingLayer
from keras.initializers import RandomUniform

def KNCN_CN_model(maxlen, max_features, embedding_size, WordEM,
                  maxlen2, max_features2, embedding_size2, WordEM2):

    rng =RandomUniform(-0.25,0.25,1337)
    left_inputs = Input(shape=(maxlen,), dtype='int32', name='left_input')
    inputs = Embedding(max_features, embedding_size, input_length=maxlen,
                       weights=[WordEM],
                     )(left_inputs)
    right_input = Input(shape=(maxlen2,), dtype='int32', name='right_input')
    right = Embedding(max_features2, embedding_size2, input_length=maxlen2,trainable=True,
                      # embeddings_initializer=rng
                      weights=[WordEM2],

                      )(right_input)

    word_level_att =AttMemoryLayer(name='watt')([right,inputs])
    right =Flatten()(right)
    word_level_att =concatenate([word_level_att,right])

    out =Dense(100,activation='relu')(word_level_att)
    out =Dropout(0.5)(out)
    out =Dense(50,activation='relu')(out)
    out =Dropout(0.5)(out)
    predict = Dense(2, activation='softmax')(out)

    model = Model(inputs=[left_inputs, right_input], outputs=predict)
    return model

def NAM_model(maxlen, max_features, embedding_size, WordEM,
              maxlen2, max_features2, embedding_size2, WordEM2):
    #拼接关系向量到词向量下面
    # rng = RandomUniform(-0.25, 0.25, 1337)
    left_inputs = Input(shape=(maxlen,), dtype='int32', name='left_input')
    inputs = Embedding(max_features, embedding_size, input_length=maxlen, weights=[WordEM],
                     )(left_inputs)
    right_input = Input(shape=(maxlen2,), dtype='int32', name='right_input')
    right = Embedding(max_features2, embedding_size2, input_length=maxlen2,trainable=True,
                      #embeddings_initializer=rng
                      weights=[WordEM2],
                      )(right_input)

    word_rel =Flatten()(right)
    emb2 =RepeatVector(maxlen)(word_rel)
    word_relation =concatenate([inputs,emb2],2)
    out =AttentivePoolingLayer(name='att')(word_relation)


    out =Dense(100,activation='relu')(out)
    out =Dropout(0.5)(out)
    out =Dense(50,activation='relu')(out)
    out =Dropout(0.5)(out)
    predict = Dense(2, activation='softmax')(out)

    model = Model(inputs=[left_inputs, right_input], outputs=predict)
    return model


def CN_CN_model(maxlen, max_features, embedding_size, WordEM,
                maxlen2, max_features2, embedding_size2, WordEM2):


    left_inputs = Input(shape=(maxlen,), dtype='int32', name='left_input')
    inputs = Embedding(max_features, embedding_size, input_length=maxlen,
                       weights=[WordEM],
                     )(left_inputs)
    right_input = Input(shape=(maxlen2,), dtype='int32', name='right_input')
    right = Embedding(max_features2, embedding_size2, input_length=maxlen2,trainable=True,


                      weights=[WordEM2],

                      )(right_input)

    word_level_att =AttentivePoolingLayer(name='att')(inputs)
    out =Dense(100,activation='relu')(word_level_att)
    out =Dropout(0.5)(out)
    out =Dense(50,activation='relu')(out)
    out =Dropout(0.5)(out)
    predict = Dense(2, activation='softmax')(out)

    model = Model(inputs=[left_inputs, right_input], outputs=predict)
    return model

def CN_KNCN_model(maxlen, max_features, embedding_size, WordEM,
                maxlen2, max_features2, embedding_size2, WordEM2):


    left_inputs = Input(shape=(maxlen,), dtype='int32', name='left_input')
    inputs = Embedding(max_features, embedding_size, input_length=maxlen,
                       weights=[WordEM],
                     )(left_inputs)
    right_input = Input(shape=(maxlen2,), dtype='int32', name='right_input')
    right = Embedding(max_features2, embedding_size2, input_length=maxlen2,trainable=True,


                      weights=[WordEM2],

                      )(right_input)
    from customize_layer import  Att_prior
    word_level_att =Att_prior(name='att')([right,inputs])
    # right =Flatten()(right)
    # out =concatenate([word_level_att,right])
    out =Dense(100,activation='relu')(word_level_att)
    out =Dropout(0.5)(out)
    out =Dense(50,activation='relu')(out)
    out =Dropout(0.5)(out)
    predict = Dense(2, activation='softmax')(out)

    model = Model(inputs=[left_inputs, right_input], outputs=predict)
    return model
