
from nlpia.loaders import get_data
df = get_data('moviedialog')

#用于存储从语料库中读取的输入文本和目标文本
input_texts, target_text =[], []

#保存输入文本和目标文本中出现过的字符
input_vocabulary = set()
output_vocabulary = set()

#特殊标记
start_token = '\t'
stop_token = '\n'

#选择需要训练的样本行数 用户自定义值或数据行数
max_training_samples = min(25000, len(df) - 1)

for input_text, target_text in zip(df.statement, df.reply):
    target_text = start_token + target_text + stop_token
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_vocabulary:
            input_vocabulary.add(char)
    for char in target_text:
        if char not in output_vocabulary:
            output_vocabulary.add(char)

#将字符集转换为排序后的字符列表 然后使用此列表生成字典
input_vocabulary = sorted(input_vocabulary)
output_vocabulary = sorted(output_vocabulary)

input_vocab_size = len(input_vocabulary)
output_vocab_size = len(output_vocabulary)

#设置输入输出序列词条的最大数量
max_encoder_seq_length = max(
    [len(txt) for txt in input_texts]
)

max_decoder_seq_length = max(
    [len(txt) for txt in target_texts]
)

input_token_index = dict([(char, i) for i, char in enumerate(input_vocabulary)])
target_token_index = dict([(char, i) for i, char in enumerate(output_vocabulary)])

reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

#构造字符级别序列编码-解码训练集
import numpy as np

#训练的张量初始化为(samples_nums, max_len_sequence, num_unique_tokens_in_vocab)
encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, input_vocab_size), dtype = 'float32')

decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, output_vocab_size), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
     for t, char in enumerate(input_text):
          encoder_input_data[i, t, input_token_index[char]] = 1
     
     for t, char in enumerate(target_text):
         decoder_input_data[i, t, target_token_index[char]] = 1
         if t > 0:
             decoder_target_data[i, t - 1, target_token_index[char]] = 1

#构造字符级序列编码-解码网络
from keras.models import Model
from keras.layers import Input, LSTM, Dense

#设置超参数
batch_size = 64
epochs = 100

num_neurons = 256

#encoder的输入维度
encoder_inputs = Input(shape = (None, input_vocab_size))
encoder = LSTM(num_neurons, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

#decoder结构
decoder_inputs = Input(shape=(None, output_vocab_size))
decoder_lstm = LSTM(num_neurons, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

decoder_dense = Dense(output_vocab_size, activation='softmax')

decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs, validation_split=0.1)

#组装序列生成模型

encoder_model = Model(encoder_inputs, encoder_states)
thought_input = [Input(shape = (num_neurons,)), Input(shape=(num_neurons,))]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=thought_input)decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    inputs = [decoder_inputs] + thought_input,
    output = [decoder_outputs] + decoder_states
)

def decode_sequence(input_seq):
    thought = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, output_vocab_size))
    target_seq[0, 0, target_token_index[stop_token]] = 1
    stop_condition = False
    generated_sequence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + thought)
        
        generated_token_idx = np.argmax(output_tokens[0, -1, :])
        generated_char = reverse_target_char_index[generated_token_idx]
        generated_sequence += generated_char
        
        if (generated_char == stop_token or len(generated_sequence) > max_decoder_seq_length):
            stop_condition = True
        target_seq = np.zeros((1, 1, output_vocab_size))
        target_seq[0, 0, generated_token_idx] = 1
        thought = [h, c]

    return generated_sequence

def response(input_text):
    input_seq = np.zeros((1, max_encoder_seq_length, input_vocab_size), dtype = 'float32')
    
    for t, char in enumerate(input_text):
        input_seq[0, t, input_token_index[char]] = 1
    
    decoded_sentence = decode_sequence(input_seq)

    print('Bot Reply (Decoded sentence):', decoded_sentence)
