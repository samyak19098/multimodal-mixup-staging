import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import *

from tqdm.auto import tqdm
from sklearn.metrics import *
import pandas as pd
import numpy as np
import math
# %matplotlib inline
import matplotlib.pyplot as plt
import gc

from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def mcc_m(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

#Hyper Parameters
batch_size = 64
learning_rate = 0.001

volatility_feedforward_size = 16
volatility_hidden_dim = 16
volatility_dropout = 0.1

movement_feedforward_size = 64
movement_hidden_dim = 32
movement_dropout = 0.0

"""# M3A Function Declaration"""

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8,**kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
        })
        return config

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1,**kwargs):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = MultiHeadSelfAttention(self.embed_dim, self.num_heads)
        self.ffn = keras.Sequential(
            [Dense(self.ff_dim, activation="relu"), Dense(self.embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim
        })
        return config

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def createModelV(emd1, emd2, heads, dimFF, dimH, drop, maxlen, maxSpeaker):
  embed_dim1 = emd1   # Embedding size for Text 
  embed_dim2 = emd2   # Embedding size for Audio
  num_heads = heads   # Number of attention heads
  ff_dim = dimFF      # Hidden layer size in feed forward network inside transformer
  hidden_dim = dimH   # Hidden layer Dimension
  dropout = drop      # Dropout

  text = Input(shape=(maxlen,embed_dim1))
  audio = Input(shape=(maxlen,embed_dim2))
  pos = Input(shape=(maxlen,embed_dim2))
  speak = Input(shape=(maxLen,maxSpeaker+1))

  newtext = TimeDistributed(Dense(62))(text)

  attentionText2 = TimeDistributed(Dense(62, activation='softmax'))(newtext)
  attentionAudio2 = TimeDistributed(Dense(62, activation='softmax'))(audio)
  attentionSum = attentionText2+attentionAudio2
  attentionText2 = attentionText2/attentionSum
  attentionAudio2 = attentionAudio2/attentionSum

  attendedText = newtext*attentionText2 
  attendedAudio = audio*attentionAudio2 

  fused = attendedText*attentionText2 + attendedAudio*attentionAudio2 + pos
  fusedSpeaker = Concatenate(axis=2)([fused,speak])

  transformer_block = TransformerBlock(embed_dim2+maxSpeaker+1, num_heads, ff_dim, 0)
  x = transformer_block(fusedSpeaker)
  x = layers.GlobalAveragePooling1D()(x)
  x = layers.Dense(hidden_dim, activation="relu")(x)
  x = layers.Dropout(dropout)(x)
  outputs = layers.Dense(1, activation="relu")(x)

  model = keras.Model(inputs=[text,audio,pos,speak], outputs=outputs)
  return model

def get_encoder(emd1, emd2, heads, dimFF, dimH, drop, maxlen, maxSpeaker):
  embed_dim1 = emd1   # Embedding size for Text 
  embed_dim2 = emd2   # Embedding size for Audio
  num_heads = heads   # Number of attention heads
  ff_dim = dimFF      # Hidden layer size in feed forward network inside transformer
  hidden_dim = dimH   # Hidden layer Dimension
  dropout = drop      # Dropout

  text = Input(shape=(maxlen,embed_dim1))
  audio = Input(shape=(maxlen,embed_dim2))
  pos = Input(shape=(maxlen,embed_dim2))
  speak = Input(shape=(maxLen,maxSpeaker+1))

  newtext = TimeDistributed(Dense(62))(text)

  attentionText1 = TimeDistributed(Dense(62, activation='softmax'))(newtext)
  attentionAudio1 = TimeDistributed(Dense(62, activation='softmax'))(audio)
  attentionText2 = TimeDistributed(Dense(62, activation='softmax'))(newtext)
  attentionAudio2 = TimeDistributed(Dense(62, activation='softmax'))(audio)
  attentionSum = attentionText2+attentionAudio2
  attentionText2 = attentionText2/attentionSum
  attentionAudio2 = attentionAudio2/attentionSum

  attendedText = newtext*attentionText2 
  attendedAudio = audio*attentionAudio2 

  fused = attendedText*attentionText2 + attendedAudio*attentionAudio2 + pos
  fusedSpeaker = Concatenate(axis=2)([fused,speak])

  return keras.Model(inputs=[text, audio, pos, speak], outputs=fusedSpeaker)

  

def createModelC(emd1, emd2, heads, dimFF, dimH, drop, maxlen, maxSpeaker):
  embed_dim1 = emd1   # Embedding size for Text 
  embed_dim2 = emd2   # Embedding size for Audio
  num_heads = heads   # Number of attention heads
  ff_dim = dimFF      # Hidden layer size in feed forward network inside transformer
  hidden_dim = dimH   # Hidden layer Dimension
  dropout = drop      # Dropout
  
  text_1 = Input(shape=(maxlen,embed_dim1))
  audio_1 = Input(shape=(maxlen,embed_dim2))
  pos_1 = Input(shape=(maxlen,embed_dim2))
  speak_1 = Input(shape=(maxLen,maxSpeaker+1))

  text_2 = Input(shape=(maxlen,embed_dim1))
  audio_2 = Input(shape=(maxlen,embed_dim2))
  pos_2 = Input(shape=(maxlen,embed_dim2))
  speak_2 = Input(shape=(maxLen,maxSpeaker+1))

  mixing_ratio = Input(shape=(1,1))

  encoder = get_encoder(emd1, emd2, heads, dimFF, dimH, drop, maxlen, maxSpeaker)
  fusedSpeaker_1 = encoder([text_1, audio_1, pos_1, speak_1])
  fusedSpeaker_2 = encoder([text_2, audio_2, pos_2, speak_2])
  fusedSpeaker = mixing_ratio * fusedSpeaker_1 + (1 - mixing_ratio) * fusedSpeaker_2
  

  transformer_block = TransformerBlock(embed_dim2+maxSpeaker+1, num_heads, ff_dim, 0)
  x = transformer_block(fusedSpeaker)
  x = layers.GlobalAveragePooling1D()(x)
  x = layers.Dense(hidden_dim, activation="relu")(x)
  x = layers.Dropout(dropout)(x)
  outputs = layers.Dense(1, activation="sigmoid")(x)

  model = keras.Model(inputs=[text_1,audio_1,pos_1,speak_1, text_2, audio_2, pos_2, speak_2, mixing_ratio], outputs=outputs)
  return model

"""# M3A Data Loading and Processing"""

YVals = pd.read_csv("Y_Volatility.csv")
files = YVals['File Name']
dates = YVals['Date']

path_to_files = 'Dataset/' # insert path with last /

X = []
maxLen = 0
index = 1
for i in tqdm(range(len(files))):
  f = files[i][:-4]
  d = str(dates[i]).split('/')
  date = d[-1] + '-'
  if(len(d[0]) == 2):
    date += d[0] + '-'
  else:
    date += '0'+d[0]+'-'
  if(len(d[1]) == 2):
    date += d[1] 
  else:
    date += '0'+d[1]
  f = f.replace('&', '_')
  folder = path_to_files + f + '_' + date+'/'
  df = pd.read_csv(folder+"Text.csv")
  df = df.drop([df.columns[0], df.columns[1]],axis=1)
  xEmb = df.to_numpy()
  X.append(xEmb)
  maxLen = max(maxLen, xEmb.shape[0])
  # print(index,f)
  index += 1

for i in tqdm(range(len(X))):
  xEmb = X[i]
  pad = maxLen-xEmb.shape[0]
  if pad != 0:
    padding = np.zeros((pad,768))
    xEmb = np.concatenate((padding,xEmb),axis=0)
  X[i] = xEmb
X_text = np.array(X)

Xspeak = []
maxSpeaker = 0
index = 1
for i in tqdm(range(len(files))):
  f = files[i][:-4]
  d = str(dates[i]).split('/')
  date = d[-1] + '-'
  if(len(d[0]) == 2):
    date += d[0] + '-'
  else:
    date += '0'+d[0]+'-'
  if(len(d[1]) == 2):
    date += d[1] 
  else:
    date += '0'+d[1]
  f = f.replace('&', '_')
  folder = path_to_files + f + '_' + date+'/'
  df = pd.read_csv(folder+"Text.csv")
  speaker = df['Speaker']
  Xspeak.append(speaker)
  maxSpeaker = max(maxSpeaker,max(speaker))

for i in tqdm(range(len(Xspeak))):
  speaker = Xspeak[i]
  s = []
  for j in range(len(speaker)):
    temp = np.zeros((maxSpeaker+1,))
    temp[speaker[j]] = 1
    s.append(temp)
  s = np.array(s)
  pad = maxLen-speaker.shape[0]
  if pad != 0:
    padding = np.zeros((pad,maxSpeaker+1))
    s = np.concatenate((padding,s),axis=0)
  Xspeak[i] = s
Xspeak = np.array(Xspeak)

X = []
maxLen = 0
index = 1
for i in tqdm(range(len(files))):
  f = files[i][:-4]
  d = str(dates[i]).split('/')
  date = d[-1] + '-'
  if(len(d[0]) == 2):
    date += d[0] + '-'
  else:
    date += '0'+d[0]+'-'
  if(len(d[1]) == 2):
    date += d[1] 
  else:
    date += '0'+d[1]
  f = f.replace('&', '_')
  folder = path_to_files + f + '_' + date+'/'
  df = pd.read_csv(folder+"Audio.csv")
  df = df.drop([df.columns[0]],axis=1)
  xEmb = df.to_numpy()
  X.append(xEmb)
  maxLen = max(maxLen, xEmb.shape[0])
  index += 1

for i in tqdm(range(len(X))):
  xEmb = X[i]
  pad = maxLen-xEmb.shape[0]
  if pad != 0:
    padding = np.zeros((pad,62))
    xEmb = np.concatenate((padding,xEmb),axis=0)
  X[i] = xEmb
X_audio = np.array(X)

for i in range(len(X)):
  for j in range(len(X[i])):
    for k in range(len(X[i][j])):
      if np.isnan(X[i][j][k]):
        X_audio[i][j][k] = 0

pos = np.zeros(X_audio.shape)
for i in tqdm(range(len(X_audio))):
  for j in range(len(X_audio[i])):
    for d in range(len(X_audio[i][j])):
      if d %2 == 0:
        p = math.sin(j/pow(10000,d/62))
        pos[i][j][d] = p
      else:
        p = math.cos(j/pow(10000,(d-1)/62))
        pos[i][j][d] = p

print(X_text.shape,X_audio.shape,Xspeak.shape,pos.shape)

trainIndex = pd.read_csv("Train_index.csv")
trainIndex = trainIndex['index']
testIndex = pd.read_csv("Test_index.csv")
testIndex = testIndex['index']

X_text_Train = X_text[trainIndex]
X_text_Test = X_text[testIndex]
X_audio_Train = X_audio[trainIndex]
X_audio_Test = X_audio[testIndex]
X_pos_Train = pos[trainIndex]
X_pos_Test = pos[testIndex]
X_speak_Train = Xspeak[trainIndex]
X_speak_Test = Xspeak[testIndex]

YVals = pd.read_csv("Y_Volatility.csv")
YT3 = YVals['vFuture3']
YT7 = YVals['vFuture7']
YT15 = YVals['vFuture15']

Ys = [YT3,YT7,YT15]
YPrint = ["Tau 3","Tau 7","Tau 15"]
# print("------------------------------ Starting training -----------------------------------")
# for i in range(3):
#   print(f"At i = {i}")
#   YTrain = Ys[i][trainIndex]
#   YTest = Ys[i][testIndex]
  
#   modelN = "ModelV "+YPrint[i]+".h5"

#   mc = tf.keras.callbacks.ModelCheckpoint(modelN, monitor='val_loss', verbose=0, save_best_only=True)
#   model = createModelV(768, 62, 3, volatility_feedforward_size, volatility_hidden_dim, volatility_dropout, maxLen,maxSpeaker)
#   model.compile(loss='mean_squared_error', optimizer=Adam(lr = learning_rate))
#   out = model.fit([X_text_Train,X_audio_Train,X_pos_Train,X_speak_Train], YTrain, batch_size=batch_size, epochs=500, validation_data=([X_text_Test,X_audio_Test,X_pos_Test,X_speak_Test],YTest), verbose=1, callbacks=[mc])
#   depen = {'MultiHeadSelfAttention': MultiHeadSelfAttention,'TransformerBlock': TransformerBlock} 
#   model = load_model(modelN, custom_objects=depen)
  
#   predTest = model.predict([X_text_Train,X_audio_Train,X_pos_Train,X_speak_Train])
#   r = mean_squared_error(YTrain,predTest)
#   print('MSE for Training Set for ',YPrint[i],': ',r)
  
#   predTest = model.predict([X_text_Test,X_audio_Test,X_pos_Test,X_speak_Test])
#   r = mean_squared_error(YTest,predTest)
#   print('MSE for Testing Set for ',YPrint[i],': ',r)
#   print()

YVals = pd.read_csv("Y_Movement.csv")
YT3 = YVals['YT3']
YT7 = YVals['YT7']
YT15 = YVals['YT15']

Ys = [YT3,YT7,YT15]
YPrint = ["Tau 3","Tau 7","Tau 15"]

print("------------------------------ Starting training 2-----------------------------------")
for i in range(3):
  print(f"At i = {i}")
  YTrain = Ys[i][trainIndex]
  YTest = Ys[i][testIndex]

  modelN = "ModelC "+YPrint[i]+".h5"

  mc = tf.keras.callbacks.ModelCheckpoint(modelN, monitor='val_accuracy', verbose=0, save_best_only=True)
  model = createModelC(768, 62, 3, movement_feedforward_size, movement_hidden_dim, movement_dropout, maxLen,maxSpeaker)
  model.compile(loss='binary_crossentropy', optimizer=Adam(lr = learning_rate), metrics=['accuracy', f1_m, mcc_m])


  print(f"Shapes: X_text_train = {X_text_Train.shape}, X_audio_Train = {X_audio_Train.shape}, X_pos_Train = {X_pos_Train.shape}, X_speak_Train = {X_speak_Train.shape}, Y_train = {YTrain.shape}")

  # x1, x2
  # x_mix = mix(x1, x2)
  # y_mix = mix(y1, y2)
  num_samples = X_text_Train.shape[0]
  perm = np.random.permutation(num_samples)
  X_text_Train_2 = X_text_Train[perm]
  X_audio_Train_2 = X_audio_Train[perm]
  X_pos_Train_2 = X_pos_Train[perm]
  X_speak_Train_2 = X_speak_Train[perm]
  YTrain_2 = YTrain.iloc[perm]
  
  mixing_ratio = np.array([np.random.beta(0.75, 0.75) for i in range(num_samples)])
  one_indices = np.random.choice(len(mixing_ratio), size=int(0.8 * len(mixing_ratio)), replace=False)
  mixing_ratio[one_indices] = 1
  
  mixing_ratio = mixing_ratio.reshape((num_samples, 1, 1))
  print(f"mr = {mixing_ratio.shape}, ytrain = {YTrain.to_numpy().shape}, ytrain2 = {YTrain_2.to_numpy().shape}")
  Y_mix = YTrain.to_numpy() * mixing_ratio.squeeze() + YTrain_2.to_numpy() * (1 - mixing_ratio.squeeze()) 
  
  num_samples_test = X_text_Test.shape[0]
  perm_test = np.random.permutation(num_samples_test)
  X_text_Test_2 = X_text_Test[perm_test]
  X_audio_Test_2 = X_audio_Test[perm_test]
  X_pos_Test_2 = X_pos_Test[perm_test]
  X_speak_Test_2 = X_speak_Test[perm_test]
  YTest_2 = YTest.iloc[perm_test]
  mixing_ratio_test = np.array([[1] for i in range(num_samples_test)])
  mixing_ratio_test = mixing_ratio_test.reshape((num_samples_test, 1, 1))
  print(f"mr = {mixing_ratio_test.shape}, ytrain = {YTest.to_numpy().shape}, ytrain2 = {YTest_2.to_numpy().shape}")
  Y_mix_Test = YTest.to_numpy() * mixing_ratio_test.squeeze() + YTest_2.to_numpy() * (1 - mixing_ratio_test.squeeze())
  
  
  
  out = model.fit([X_text_Train,X_audio_Train,X_pos_Train,X_speak_Train, X_text_Train_2, X_audio_Train_2, X_pos_Train_2, X_speak_Train_2, mixing_ratio], Y_mix, batch_size=batch_size, epochs=150, validation_data=([X_text_Test,X_audio_Test,X_pos_Test,X_speak_Test, X_text_Test_2,X_audio_Test_2,X_pos_Test_2,X_speak_Test_2, mixing_ratio_test],Y_mix_Test), verbose=1, callbacks=[mc])
  depen = {'MultiHeadSelfAttention': MultiHeadSelfAttention,'TransformerBlock': TransformerBlock} 
  model = load_model(modelN, custom_objects=depen)
  
  mixing_ratio_ones_train = np.array([[np.random.beta(0.75, 0.75)] for i in range(num_samples)])
  mixing_ratio_ones_train = mixing_ratio_ones_train.reshape((num_samples, 1, 1))
  mixing_ratio_ones_test = np.array([[1] for i in range(num_samples_test)])
  mixing_ratio_ones_test = mixing_ratio_ones_test.reshape((num_samples_test, 1, 1))
  
  predTest = model.predict([X_text_Train,X_audio_Train,X_pos_Train,X_speak_Train, X_text_Train, X_audio_Train, X_pos_Train, X_speak_Train, mixing_ratio_ones_train]).round()
  mcc = matthews_corrcoef(YTrain, predTest)
  f1 = f1_score(YTrain,predTest)
  print('F1 for Training Set for ',YPrint[i],': ',f1)
  print('MCC for Training Set for ',YPrint[i],': ',mcc)  
  
  predTest = model.predict([X_text_Test,X_audio_Test,X_pos_Test,X_speak_Test, X_text_Test,X_audio_Test,X_pos_Test,X_speak_Test, mixing_ratio_ones_test]).round()
  mcc = matthews_corrcoef(YTest, predTest)
  f1 = f1_score(YTest,predTest)
  print('F1 for Testing Set for ',YPrint[i],': ',f1)
  print('MCC for Testing Set for ',YPrint[i],': ',mcc)
  print()