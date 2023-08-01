import numpy as np
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, Masking, Dropout, Bidirectional, Concatenate, Flatten
from keras.models import Model

def createMdrmLstm(maxlen, input_shape, output_shape, last_layer_activation, flatten=False):
    input_data = Input(shape=(maxlen, input_shape))
    masked = Masking(mask_value =0)(input_data)
    lstm = Bidirectional(LSTM(300, activation='tanh', return_sequences = True, dropout=0.6))(masked)
    inter = Dropout(0.9)(lstm)
    if flatten:
        flattened = Flatten()(inter)
        inter1 = Dense(output_shape, activation=last_layer_activation)(flattened)
    else:
        inter1 = TimeDistributed(Dense(output_shape,activation=last_layer_activation))(inter)
    model = Model(input_data, inter1)

    return model

def createMdrm(maxlen, audio_shape, text_shape):
    audio = Input(shape=(maxlen, audio_shape))
    text = Input(shape=(maxlen, text_shape))

    lstm_audio = createMdrmLstm(maxlen, audio_shape, 100, 'tanh')
    lstm_text = createMdrmLstm(maxlen, text_shape, 100, 'tanh')
    concatenator = Concatenate(axis=2)
    lstm_combined = createMdrmLstm(maxlen, 200, 1, 'sigmoid', flatten=True)

    audio_features = lstm_audio(audio)
    text_features = lstm_text(text)
    combined_features = concatenator([audio, text])
    logits = lstm_combined(combined_features)

    model = Model(inputs=[audio, text], outputs=[logits])

    return model