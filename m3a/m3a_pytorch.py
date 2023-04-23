import torch
import torch.nn as nn

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        # Assuming x.shape = (batch_size, seq_len, input_size)
        # We want to apply the module to each timestep independently
        output = []
        for i in range(x.size(1)):
            output.append(self.module(x[:, i, :]))
        # Concatenate the outputs along the sequence length dimension
        return torch.stack(output, dim=1)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.text_linear = nn.Linear(emb1, 62)
        self.attention_text_layer = TimeDistributed(
            nn.Sequential(
                nn.Linear(62, 62),
                nn.Softmax()
            )
        )
        self.attention_audio_layer = TimeDistributed(
            nn.Sequential(
                nn.Linear(62, 62),
                nn.Softmax()
            )
        )
    
    def forward(self, text, audio, pos, speak):
        text = self.text_linear(text)
        attention_text = self.attention_text_layer(text)
        attention_audio = self.attention_audio_layer(audio)
        attention_sum = attention_text + attention_audio
        
        attention_text = attention_text / attention_sum
        attention_audio = attention_audio / attention_sum

        attended_text = text * attention_text
        attended_audio = audio * attention_audio
        fused = attention_text + attended_audio

class ClassificationModel(nn.Module):
    def __init__(self, emb1, emb2, heads, dimFF, dimH, drop, maxlen, maxSpeaker):
        self.embed_dim1 = emb1   # Embedding size for Text 
        self.embed_dim2 = emb2   # Embedding size for Audio
        self.num_heads = heads   # Number of attention heads
        self.ff_dim = dimFF      # Hidden layer size in feed forward network inside transformer
        self.hidden_dim = dimH   # Hidden layer Dimension
        self.dropout = drop      # Dropout

        self.text_linear = nn.Linear(emb1, 62)
        self.attention_text = nn.Sequential(
            nn.Linear(62, 62),
            nn.Softmax()
        )
        self.attention_audio = nn.Sequential(
            nn.Linear(62, 62),
            nn.Softmax()
        )

    def forward(
            self,
            text_1,
            audio_1,
            pos_1,
            speak_1,
            text_2,
            audio_2,
            pos_2,
            speak_2,
            mixing_ratio
    ):
        attention_text_1 = 