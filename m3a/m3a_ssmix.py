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
import argparse
import sys
import wandb
import argparse
import datetime

ap = argparse.ArgumentParser()
ap.add_argument('--tau', type=int, default=0, help='0 for 3, 1 for 7 and 2 for 15')
ap.add_argument('--threshold', type=float, default=0.7, help='Saliency threshold')
ap.add_argument('--loss_original_coef', type=float, default=0.7)
ap.add_argument('--loss_intra_coef', type=float, default=0.15)
ap.add_argument('--loss_inter_coef', type=float, default=0.15)
ap.add_argument('--bs', type=int, default=64, help='Batch size')
ap.add_argument('--num_epochs', type=int, default=200, help='Number of epochs')
ap.add_argument('--lr', type=float, default=0.001, help='Learning rate')
ap.add_argument('--lam_inter', type=float, default=0.2)
args = vars(ap.parse_args())

timestamp = str(datetime.datetime.now())
wandb.init(
	project='ssmix',
	name=timestamp,
	config=args
)

print(args)

# Hyper Parameters
batch_size = args['bs']
learning_rate = args['lr']

volatility_feedforward_size = 16
volatility_hidden_dim = 16
volatility_dropout = 0.1

movement_feedforward_size = 64
movement_hidden_dim = 32
movement_dropout = 0.0

"""# M3A Function Declaration"""


class MultiHeadSelfAttention(layers.Layer):
		def __init__(self, embed_dim, num_heads=8, **kwargs):
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
				config.update(
						{
								"embed_dim": self.embed_dim,
								"num_heads": self.num_heads,
						}
				)
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
		def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
				self.embed_dim = embed_dim
				self.num_heads = num_heads
				self.ff_dim = ff_dim
				super(TransformerBlock, self).__init__(**kwargs)
				self.att = MultiHeadSelfAttention(self.embed_dim, self.num_heads)
				self.ffn = keras.Sequential(
						[
								Dense(self.ff_dim, activation="relu"),
								Dense(self.embed_dim),
						]
				)
				self.layernorm1 = LayerNormalization(epsilon=1e-6)
				self.layernorm2 = LayerNormalization(epsilon=1e-6)
				self.dropout1 = Dropout(rate)
				self.dropout2 = Dropout(rate)

		def get_config(self):
				config = super().get_config().copy()
				config.update(
						{
								"embed_dim": self.embed_dim,
								"num_heads": self.num_heads,
								"ff_dim": self.ff_dim,
						}
				)
				return config

		def call(self, inputs, training):
				attn_output = self.att(inputs)
				attn_output = self.dropout1(attn_output, training=training)
				out1 = self.layernorm1(inputs + attn_output)
				ffn_output = self.ffn(out1)
				ffn_output = self.dropout2(ffn_output, training=training)
				return self.layernorm2(out1 + ffn_output)


def createModelV(emd1, emd2, heads, dimFF, dimH, drop, maxlen, maxSpeaker):
		embed_dim1 = emd1  # Embedding size for Text
		embed_dim2 = emd2  # Embedding size for Audio
		num_heads = heads  # Number of attention heads
		ff_dim = dimFF  # Hidden layer size in feed forward network inside transformer
		hidden_dim = dimH  # Hidden layer Dimension
		dropout = drop  # Dropout

		text = Input(shape=(maxlen, embed_dim1))
		audio = Input(shape=(maxlen, embed_dim2))
		pos = Input(shape=(maxlen, embed_dim2))
		speak = Input(shape=(maxLen, maxSpeaker + 1))

		newtext = TimeDistributed(Dense(62))(text)

		attentionText2 = TimeDistributed(Dense(62, activation="softmax"))(newtext)
		attentionAudio2 = TimeDistributed(Dense(62, activation="softmax"))(audio)
		attentionSum = attentionText2 + attentionAudio2
		attentionText2 = attentionText2 / attentionSum
		attentionAudio2 = attentionAudio2 / attentionSum

		attendedText = newtext * attentionText2
		attendedAudio = audio * attentionAudio2

		fused = attendedText * attentionText2 + attendedAudio * attentionAudio2 + pos
		fusedSpeaker = Concatenate(axis=2)([fused, speak])

		transformer_block = TransformerBlock(
				embed_dim2 + maxSpeaker + 1, num_heads, ff_dim, 0
		)
		x = transformer_block(fusedSpeaker)
		x = layers.GlobalAveragePooling1D()(x)
		x = layers.Dense(hidden_dim, activation="relu")(x)
		x = layers.Dropout(dropout)(x)
		outputs = layers.Dense(1, activation="relu")(x)

		model = keras.Model(inputs=[text, audio, pos, speak], outputs=outputs)
		return model


def createModelC(emd1, emd2, heads, dimFF, dimH, drop, maxlen, maxSpeaker):
		embed_dim1 = emd1  # Embedding size for Text
		embed_dim2 = emd2  # Embedding size for Audio
		num_heads = heads  # Number of attention heads
		ff_dim = dimFF  # Hidden layer size in feed forward network inside transformer
		hidden_dim = dimH  # Hidden layer Dimension
		dropout = drop  # Dropout

		text = Input(shape=(maxlen, embed_dim1))
		audio = Input(shape=(maxlen, embed_dim2))
		pos = Input(shape=(maxlen, embed_dim2))
		speak = Input(shape=(maxLen, maxSpeaker + 1))

		newtext = TimeDistributed(Dense(62))(text)

		# attentionText1 = TimeDistributed(Dense(62, activation="softmax"))(newtext)
		# attentionAudio1 = TimeDistributed(Dense(62, activation="softmax"))(audio)
		attentionText2 = TimeDistributed(Dense(62, activation="softmax"))(newtext)
		attentionAudio2 = TimeDistributed(Dense(62, activation="softmax"))(audio)
		attentionSum = attentionText2 + attentionAudio2
		attentionText2 = attentionText2 / attentionSum
		attentionAudio2 = attentionAudio2 / attentionSum

		attendedText = newtext * attentionText2
		attendedAudio = audio * attentionAudio2

		fused = attendedText*attentionText2 + attendedAudio*attentionAudio2 + pos
		fusedSpeaker = Concatenate(axis=2)([fused, speak])

		# Insert mixup

		transformer_block = TransformerBlock(
				embed_dim2 + maxSpeaker + 1, num_heads, ff_dim, 0
		)
		x = transformer_block(fusedSpeaker)
		x = layers.GlobalAveragePooling1D()(x)
		x = layers.Dense(hidden_dim, activation="relu")(x)
		x = layers.Dropout(dropout)(x)
		outputs = layers.Dense(1, activation="sigmoid")(x)

		model = keras.Model(inputs=[text, audio, pos, speak], outputs=[outputs])
		return model


"""# M3A Data Loading and Processing"""


# parser = argparse.ArgumentParser(description="M3A")
# parser.add_argument("--run_num", type=str, help="Run number", required=True)
# args = parser.parse_args()
# print(args)


YVals = pd.read_csv("Y_Volatility.csv")
files = YVals["File Name"]
dates = YVals["Date"]

path_to_files = "Dataset/"  # insert path with last /

# X = []
# maxLen = 0
# index = 1
# for i in tqdm(range(len(files))):
# 		f = files[i][:-4]
# 		d = str(dates[i]).split("/")
# 		date = d[-1] + "-"
# 		if len(d[0]) == 2:
# 				date += d[0] + "-"
# 		else:
# 				date += "0" + d[0] + "-"
# 		if len(d[1]) == 2:
# 				date += d[1]
# 		else:
# 				date += "0" + d[1]
# 		f = f.replace("&", "_")
# 		folder = path_to_files + f + "_" + date + "/"
# 		df = pd.read_csv(folder + "Text.csv")
# 		df = df.drop([df.columns[0], df.columns[1]], axis=1)
# 		xEmb = df.to_numpy()
# 		X.append(xEmb)
# 		maxLen = max(maxLen, xEmb.shape[0])
# 		# print(index,f)
# 		index += 1

# for i in tqdm(range(len(X))):
# 		xEmb = X[i]
# 		pad = maxLen - xEmb.shape[0]
# 		if pad != 0:
# 				padding = np.zeros((pad, 768))
# 				xEmb = np.concatenate((padding, xEmb), axis=0)
# 		X[i] = xEmb
# X_text = np.array(X)

# Xspeak = []
# maxSpeaker = 0
# index = 1
# for i in tqdm(range(len(files))):
# 		f = files[i][:-4]
# 		d = str(dates[i]).split("/")
# 		date = d[-1] + "-"
# 		if len(d[0]) == 2:
# 				date += d[0] + "-"
# 		else:
# 				date += "0" + d[0] + "-"
# 		if len(d[1]) == 2:
# 				date += d[1]
# 		else:
# 				date += "0" + d[1]
# 		f = f.replace("&", "_")
# 		folder = path_to_files + f + "_" + date + "/"
# 		df = pd.read_csv(folder + "Text.csv")
# 		speaker = df["Speaker"]
# 		Xspeak.append(speaker)
# 		maxSpeaker = max(maxSpeaker, max(speaker))

# for i in tqdm(range(len(Xspeak))):
# 		speaker = Xspeak[i]
# 		s = []
# 		for j in range(len(speaker)):
# 				temp = np.zeros((maxSpeaker + 1,))
# 				temp[speaker[j]] = 1
# 				s.append(temp)
# 		s = np.array(s)
# 		pad = maxLen - speaker.shape[0]
# 		if pad != 0:
# 				padding = np.zeros((pad, maxSpeaker + 1))
# 				s = np.concatenate((padding, s), axis=0)
# 		Xspeak[i] = s
# Xspeak = np.array(Xspeak)

# X = []
# maxLen = 0
# index = 1
# for i in tqdm(range(len(files))):
# 		f = files[i][:-4]
# 		d = str(dates[i]).split("/")
# 		date = d[-1] + "-"
# 		if len(d[0]) == 2:
# 				date += d[0] + "-"
# 		else:
# 				date += "0" + d[0] + "-"
# 		if len(d[1]) == 2:
# 				date += d[1]
# 		else:
# 				date += "0" + d[1]
# 		f = f.replace("&", "_")
# 		folder = path_to_files + f + "_" + date + "/"
# 		df = pd.read_csv(folder + "Audio.csv")
# 		df = df.drop([df.columns[0]], axis=1)
# 		xEmb = df.to_numpy()
# 		X.append(xEmb)
# 		maxLen = max(maxLen, xEmb.shape[0])
# 		index += 1

# for i in tqdm(range(len(X))):
# 		xEmb = X[i]
# 		pad = maxLen - xEmb.shape[0]
# 		if pad != 0:
# 				padding = np.zeros((pad, 62))
# 				xEmb = np.concatenate((padding, xEmb), axis=0)
# 		X[i] = xEmb
# X_audio = np.array(X)
# for i in range(len(X)):
# 		for j in range(len(X[i])):
# 				for k in range(len(X[i][j])):
# 						if np.isnan(X[i][j][k]):
# 				int				X_audio[i][j][k] = 0

# pos = np.zeros(X_audio.shape)
# for i in tqdm(range(len(X_audio))):
# 		for j in range(len(X_audio[i])):
# 				for d in range(len(X_audio[i][j])):
# 						if d % 2 == 0:
# 								p = math.sin(j / pow(10000, d / 62))
# 								pos[i][j][d] = p
# 						else:
# 								p = math.cos(j / pow(10000, (d - 1) / 62))
# 								pos[i][j][d] = p

# print(X_text.shape, X_audio.shape, Xspeak.shape, pos.shape)

trainIndex = pd.read_csv("Train_index.csv")
trainIndex = trainIndex["index"]
testIndex = pd.read_csv("Test_index.csv")
testIndex = testIndex["index"]

# X_text_Train = X_text[trainIndex]
# X_text_Test = X_text[testIndex]
# X_audio_Train = X_audio[trainIndex]
# X_audio_Test = X_audio[testIndex]
# X_pos_Train = pos[trainIndex]
# X_pos_Test = pos[testIndex]
# X_speak_Train = Xspeak[trainIndex]
# X_speak_Test = Xspeak[testIndex]

X_text_Train = np.load('processed_data/x_text_train.npy')
X_text_Test = np.load('processed_data/x_text_test.npy')
X_audio_Train = np.load('processed_data/x_audio_train.npy')
X_audio_Test = np.load('processed_data/x_audio_test.npy')
X_pos_Train = np.load('processed_data/x_pos_train.npy')
X_pos_Test = np.load('processed_data/x_pos_test.npy')
X_speak_Train = np.load('processed_data/x_speak_train.npy')
X_speak_Test = np.load('processed_data/x_speak_test.npy')

maxLen = 284
maxSpeaker = 30

YVals = pd.read_csv("Y_Volatility.csv")
YT3 = YVals["vFuture3"]
YT7 = YVals["vFuture7"]
YT15 = YVals["vFuture15"]

Ys = [YT3, YT7, YT15]
YPrint = ["Tau 3", "Tau 7", "Tau 15"]
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
YT3 = YVals["YT3"]
YT7 = YVals["YT7"]
YT15 = YVals["YT15"]

Ys = [YT3, YT7, YT15]
YPrint = ["Tau 3", "Tau 7", "Tau 15"]

print(
		"------------------------------ Starting training 2-----------------------------------"
)

def intra_mix(x1, x2, sal1, sal2, threshold, lam):
	threshold = 1 - threshold
	mixed = np.zeros(x1.shape)
	for i in range(x1.shape[0]):
		for j in range(x1.shape[1]):
			sample1 = x1[i, j]
			sample2 = x2[i, j]

			sample1_indices = sal1[i, j].argsort()[:int(threshold * sample1.shape[0])]
			sample2_indices = sal2[i, j].argsort()[int(-threshold * sample2.shape[0]):]

			sample1[sample1_indices] = 0
			sample2[sample2_indices] = 0
			
			mixed[i, j] = lam * sample1 + (1 - lam) * sample2

	return mixed


def inter_mix(x1, x2, sal1, sal2, span_ratio):
	
    # print("\n---------------------------")
    span_length = int(span_ratio * x1.shape[1])
    utterance_sal1 = np.sum(sal1, axis=2)
    utterance_sal2 = np.sum(sal2, axis=2)
    
    most_salient1 = [[0, span_length - 1] for i in range(x1.shape[0])]
    max_sum1 = [utterance_sal1[i][:span_length].sum() for i in range(x1.shape[0])]
    cur_sum1 = [utterance_sal1[i][:span_length].sum() for i in range(x1.shape[0])]
    
    least_salient2 = [[0, span_length - 1] for i in range(x1.shape[0])]
    min_sum2 = [utterance_sal2[i][:span_length].sum() for i in range(x1.shape[0])]
    cur_sum2 = [utterance_sal2[i][:span_length].sum() for i in range(x1.shape[0])]
    
    # print(most_salient1, max_sum1, cur_sum1)
    # print("\n\n-------------\n\n")
    
    # print(f"Cumulative saliency shape: s1 = {utterance_sal1.shape}, s2 = {utterance_sal2.shape}")
    # # inp = input()
    # print(x1)
    # print(x2)
    
    for i in range(x1.shape[0]):
        for j in range(1, x1.shape[1] - span_length):
            # print("hey1")
            # print(cur_sum1[i], utterance_sal1[i, j - 1], utterance_sal1[i, j - 1 + span_length])
            # print("hey2")
            new_sum1 = cur_sum1[i] - utterance_sal1[i, j - 1] + utterance_sal1[i, j - 1 + span_length]
            new_sum2 = cur_sum2[i] - utterance_sal2[i, j - 1] + utterance_sal2[i, j - 1 + span_length]
            # print("here")
            # print(new_sum1, max_sum1[i])
            # print("done")
            if new_sum1 > max_sum1[i]:
                max_sum1[i] = new_sum1
                most_salient1[i] = [j, j - 1 + span_length]
            if new_sum2 < min_sum2[i]:
                min_sum2[i] = new_sum2
                least_salient2[i] = [j, j - 1 + span_length]
                
            cur_sum1[i] = new_sum1
            cur_sum2[i] = new_sum2
    
    # print(most_salient1)
    # print(least_salient2)
    
    mixed = x2.copy()
    for i in range(x1.shape[0]):
        mixed[i][least_salient2[i][0]:least_salient2[i][1] + 1, :] = x1[i][most_salient1[i][0]:most_salient1[i][1] + 1, :]
    
    # print("\n\n -----X---------------X---------------------X-------------X----------")
    # print(mixed)
    
    return mixed

def custom_training(model, train_set, X_text_Test, X_audio_Test, X_pos_Test, X_speak_Test, YTest):
	loss_fn = tf.keras.losses.BinaryCrossentropy(
			from_logits=False,
			name="binary_crossentropy",
	)
	optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
	epochs = args['num_epochs']
	best_report = None
	best_mcc = None
	for epoch in range(epochs):
		running_loss, running_loss_1, running_loss_2_intra, running_loss_2_inter = 0, 0, 0, 0
		total = 0
		for idx, (text, audio, pos, speak, label) in enumerate(train_set):
			with tf.GradientTape() as super_tape:
				with tf.GradientTape(persistent=True) as tape:
					tape.watch(audio)
					tape.watch(text)
					logits = model(inputs=[text, audio, pos, speak], training=True)
					loss_value_1 = loss_fn(label, logits)

				grads_audio = tape.gradient(loss_value_1, audio)
				grads_text = tape.gradient(loss_value_1, text)

				saliency_audio = np.abs(grads_audio.numpy())
				saliency_text = np.abs(grads_text.numpy())
				
				permutation = np.random.permutation(audio.shape[0])

				
				lam = np.random.beta(0.5, 0.5)
    
				temp1 = np.sum(audio, axis=2)
				temp2 = np.count_nonzero(temp1, axis=1)
				# lam_not = np.random.beta(0.5, 0.5)
				lam_not = args['lam_inter']
				span_len = lam_not * 284
				lam_inter = 1 - (lam_not * (284 / (temp2 + 1))) #adding 1 for smoothening
				
    
				audio_mixed_intra = intra_mix(audio.numpy(), audio.numpy()[permutation], saliency_audio, saliency_audio[permutation], args['threshold'], lam)
				text_mixed_intra = intra_mix(text.numpy(), text.numpy()[permutation], saliency_text, saliency_text[permutation], args['threshold'], lam)
				audio_mixed_intra = tf.convert_to_tensor(audio_mixed_intra)
				text_mixed_intra = tf.convert_to_tensor(text_mixed_intra)
    
				audio_mixed_inter = inter_mix(audio.numpy(), audio.numpy()[permutation], saliency_audio, saliency_audio[permutation], span_len)
				text_mixed_inter = inter_mix(text.numpy(), text.numpy()[permutation], saliency_text, saliency_text[permutation], span_len)
				audio_mixed_inter = tf.convert_to_tensor(audio_mixed_inter)
				text_mixed_inter = tf.convert_to_tensor(text_mixed_inter)
    
				label = tf.cast(label, tf.float32)
				label_mixed_intra = tf.math.scalar_mul(lam, label) + tf.math.scalar_mul(1 - lam, tf.gather(label, tf.convert_to_tensor(permutation)))
				speak = tf.cast(speak, tf.float32)
				speak_mixed_intra = tf.math.scalar_mul(lam, speak) + tf.math.scalar_mul(1 - lam, tf.gather(speak, tf.convert_to_tensor(permutation)))

				label_mixed_inter = tf.multiply(tf.constant(lam_inter, dtype=tf.float32), label) + tf.multiply(tf.constant(1 - lam_inter, dtype=tf.float32), tf.gather(label, tf.convert_to_tensor(permutation)))
				speak_mixed_inter = tf.math.scalar_mul(lam_not, speak) + tf.math.scalar_mul(1 - lam_not, tf.gather(speak, tf.convert_to_tensor(permutation)))
				super_tape.watch(audio_mixed_intra)
				super_tape.watch(text_mixed_intra)
				super_tape.watch(audio_mixed_inter)
				super_tape.watch(text_mixed_inter)
				super_tape.watch(label_mixed_intra)
				super_tape.watch(speak_mixed_intra)
				super_tape.watch(label_mixed_inter)
				super_tape.watch(speak_mixed_inter)

				logits_intra = model(inputs=[text_mixed_intra, audio_mixed_intra, pos, speak_mixed_intra], training=True)
				loss_value_2_intra = loss_fn(label_mixed_intra, logits_intra)
				logits_inter = model(inputs=[text_mixed_inter, audio_mixed_inter, pos, speak_mixed_inter], training=True)
				loss_value_2_inter = loss_fn(label_mixed_inter, logits_inter)
    
				loss_value = args['loss_original_coef'] * loss_value_1 + args['loss_intra_coef'] * loss_value_2_intra + args['loss_inter_coef'] * loss_value_2_inter


				running_loss_1 += loss_value_1 * audio.shape[0]
				running_loss_2_intra += loss_value_2_intra * audio.shape[0]
				running_loss_2_inter += loss_value_2_inter * audio.shape[0]
				running_loss += loss_value * audio.shape[0]
				total += audio.shape[0]
			
			grads = super_tape.gradient(loss_value, model.trainable_weights)
			optimizer.apply_gradients(zip(grads, model.trainable_weights))

			print(f"Epoch {epoch}: (iter {idx}) train_loss={float(loss_value)}")
		
		running_loss_1 /= total
		running_loss_2_intra /= total
		running_loss_2_inter /= total
		running_loss /= total


		predTest = model.predict(
				[X_text_Test, X_audio_Test, X_pos_Test, X_speak_Test]
		).round()
		mcc = matthews_corrcoef(YTest, predTest)
		f1 = f1_score(YTest, predTest)
		print("F1 for Testing Set for ", YPrint[i], ": ", f1)
		print("MCC for Testing Set for ", YPrint[i], ": ", mcc)
		report = classification_report(YTest, predTest, output_dict=True)
		if best_report is None:
			best_report = report
		elif best_report['weighted avg']['f1-score'] < report['weighted avg']['f1-score']:
			best_report = report
			best_mcc = mcc
		print(f"Weighted f1 score = {best_report['weighted avg']['f1-score']}")
		print()

		wandb.log({
			'Original loss': running_loss_1,
			'Intra loss': running_loss_2_intra,
			'Inter loss': running_loss_2_inter,
			'Total loss': running_loss,
			'F1-score': report['weighted avg']['f1-score'],
			'MCC': mcc
		})
	wandb.log({
		'Best F1-score': best_report['weighted avg']['f1-score'],
		'Best MCC': best_mcc
	})



i = args['tau']
print(f"At i = {i}")
YTrain = Ys[i][trainIndex]
YTest = Ys[i][testIndex]

train_set = tf.data.Dataset.from_tensor_slices(
		(X_text_Train, X_audio_Train, X_pos_Train, X_speak_Train, YTrain)
)
train_set = train_set.batch(batch_size=batch_size)
test_set = tf.data.Dataset.from_tensor_slices(
		(X_text_Test, X_audio_Test, X_pos_Test, X_speak_Test, YTest)
)

print(f"shapes: Y_train = {YTrain.shape}, YTest = {YTest.shape}")
modelN = f"saved-ModelC " + YPrint[i] + ".h5"

mc = tf.keras.callbacks.ModelCheckpoint(
		modelN, monitor="val_accuracy", verbose=0, save_best_only=True
)
model = createModelC(
		768,
		62,
		3,
		movement_feedforward_size,
		movement_hidden_dim,
		movement_dropout,
		maxLen,
		maxSpeaker,
)

custom_training(model, train_set, X_text_Test, X_audio_Test, X_pos_Test, X_speak_Test, YTest)

# model.compile(loss='binary_crossentropy', optimizer=Adam(lr = learning_rate), metrics=['accuracy'])

# out = model.fit([X_text_Train,X_audio_Train,X_pos_Train,X_speak_Train], YTrain, batch_size=batch_size, epochs=200, validation_data=([X_text_Test,X_audio_Test,X_pos_Test,X_speak_Test],YTest), verbose=1, callbacks=[mc])
# depen = {
# 		"MultiHeadSelfAttention": MultiHeadSelfAttention,
# 		"TransformerBlock": TransformerBlock,
# }
# model = load_model(modelN, custom_objects=depen)


