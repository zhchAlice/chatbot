# -*- coding: utf-8 -*-
# @Time    : 2018/2/7 10:26
# @Author  : Alice
# @File    : seq2seq.py
# @Desc    :
from gensim.models import word2vec
import numpy as np
import tflearn
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
import random


class Seq2Seq(object):
  def __init__(self, word_vector_model_path, seq2seq_model_path, input_file, word_vec_dim=200, max_seq_len=16):
    """
    :param word_vector_model_path: 已经训练好的word2vec词向量模型路径
    :param seq2seq_model_path: seq2seq模型路径
    :param input_file: 已经处理好的问答对源文件
    :param word_vec_dim: 词向量长度
    :param max_seq_len: 最大序列长度
    """
    self.word_vector_model_path = word_vector_model_path
    self.input_file = input_file
    self.word_vec_dim = word_vec_dim
    self.max_seq_len = max_seq_len
    self.seq2seq_model_path = seq2seq_model_path
    self.question_seqs = []  # 问题序列集
    self.answer_seqs = []  # 回答序列集
    self.word_vector_dict = {}

  def load_word_vector_dict(self):
    model = word2vec.Word2Vec.load(self.word_vector_model_path)
    vocab = model.wv.vocab
    word_vector = {}
    for word in vocab:
      self.word_vector_dict[word] = model[word]

  def init_seq(self):
    """
    初始化问答词向量序列
    """
    if not self.word_vector_dict:
      self.load_word_vector_dict()

    file_object = open(self.input_file, 'r', encoding='utf-8')
    while True:
      line = file_object.readline()
      if line:
        line_pair = line.split("|")
        line_question_words = line_pair[0].split(" ")
        line_answer_words = line_pair[1].split(" ")
        question_seq = []
        answer_seq = []
        for word in line_question_words:
          if word in self.word_vector_dict:
            question_seq.append(self.word_vector_dict[word])
        for word in line_answer_words:
          if word in self.word_vector_dict:
            answer_seq.append(self.word_vector_dict[word])
        self.question_seqs.append(question_seq)
        self.answer_seqs.append(answer_seq)
      else:
        break
    file_object.close()

  def generate_trainig_data(self):
    if not self.question_seqs:
      self.init_seq()
    # xy_data = []
    # y_data = []
    train_XY=np.empty(shape=[0,32,200])
    train_Y=np.empty(shape=[0,17,200])
    print(len(self.question_seqs))
    for i in range(len(self.question_seqs)):
      question_seq = self.question_seqs[i]
      answer_seq = self.answer_seqs[i]
      # 输入序列长度补齐为max_seq_len
      if len(question_seq) < self.max_seq_len and len(answer_seq) < self.max_seq_len:
        seq_xy = [np.zeros(self.word_vec_dim)] * (self.max_seq_len - len(question_seq)) + list(reversed(question_seq))
        seq_y = answer_seq + [np.zeros(self.word_vec_dim)] * (self.max_seq_len - len(answer_seq))
        seq_xy = seq_xy + seq_y
        seq_y = [np.ones(self.word_vec_dim)] + seq_y
        # xy_data.append(seq_xy)
        # y_data.append(seq_y)
        train_XY = np.append(train_XY,[seq_xy],axis=0)
        train_Y = np.append(train_Y,[seq_y],axis=0)
        return train_XY,train_Y
    #     test_xy = np.array(train_XY)
    #     test_y = np.array(train_Y)
    #     print(test_xy.shape)
    #     print(test_y.shape)
    # return np.array(xy_data), np.array(y_data)

  def seq2seq_model(self):
    # 为输入的样本数据申请变量空间，每个样本最多包含max_seq_len*2个词（包含qustion和answer），每个词用word_vec_dim维浮点数表示
    input_data = tflearn.input_data(shape=[None, self.max_seq_len * 2, self.word_vec_dim], name="XY")
    # 从输入的所有样本数据的词序列中切出前max_seq_len个，也就是question句子部分的词向量作为编码器的输入
    encoder_inputs = tf.slice(input_data, [0, 0, 0], [-1, self.max_seq_len, self.word_vec_dim], name="enc_in")
    # 再取出后max_seq_len-1个，也就是answer句子部分的词向量作为解码器的输入，这里只取了max_seq_len-1个，因为要在前面拼上一组
    # GO标识来告诉解码器要开始解码了
    decoder_inputs_tmp = tf.slice(input_data, [0, self.max_seq_len, 0], [-1, self.max_seq_len - 1, self.word_vec_dim],
                                  name="dec_in_tmp")
    go_inputs = tf.ones_like(decoder_inputs_tmp)
    go_inputs = tf.slice(go_inputs, [0, 0, 0], [-1, 1, self.word_vec_dim])
    # 插入GO标识作为解码器的第一个输入
    decoder_inputs = tf.concat([go_inputs, decoder_inputs_tmp], 1, name="dec_in")
    # 开始编码过程，返回的encoder_output_tensor展开成tflearn.regression回归可以识别的形如(?,1,200)的向量
    (encoder_output_tensor, states) = tflearn.lstm(encoder_inputs, self.word_vec_dim, return_state=True,
                                                   scope="encoder_lstm")
    encoder_output_sequence = tf.stack([encoder_output_tensor], axis=1)
    # 获取decoder的第一个字符,即GO标识
    first_dec_input = tf.slice(decoder_inputs, [0, 0, 0], [-1, 1, self.word_vec_dim])
    # 将GO标识输入到解码器中，解码器的state初始化为编码器生成的states，这里的scope='decoder_lstm'是为了下面重用同一个解码器
    decoder_output_tensor = tflearn.lstm(first_dec_input, self.word_vec_dim, initial_state=states, return_state=False,
                                         reuse=False, scope="decoder_lstm")
    # 暂时先将解码器的第一个输出存到decoder_output_sequence_list中供最后一起输出
    decoder_output_sequence_single = tf.stack([decoder_output_tensor], axis=1)
    decoder_output_sequence_list = [decoder_output_tensor]
    # 接下来我们循环max_seq_len-1次，不断取decoder_inputs的一个个词向量作为下一轮解码器输入，并将结果添加到
    # decoder_output_sequence_list中，这里面的reuse=True,scope="decoder_lstm"说明和上面第一次解码用的是同一个lstm层
    for i in range(self.max_seq_len - 1):
      next_dec_input = tf.slice(decoder_inputs, [0, i, 0], [-1, 1, self.word_vec_dim])
      decoder_output_tensor = tflearn.lstm(next_dec_input, self.word_vec_dim, return_seq=False, reuse=True,
                                           scope="decoder_lstm")
      decoder_output_sequence_single = tf.stack([decoder_output_tensor], axis=1)
      decoder_output_sequence_list.append(decoder_output_tensor)
    # 下面我们把编码器第一个输出和解码器所有输出拼接起来，作为tflearn.regression回归的输入
    decode_output_sequence = tf.stack(decoder_output_sequence_list, axis=1)
    real_output_sequence = tf.concat([encoder_output_sequence, decode_output_sequence], axis=1)
    net = tflearn.regression(real_output_sequence, optimizer='sgd', learning_rate=0.1, loss='mean_square')
    model = tflearn.DNN(net, tensorboard_verbose=3, tensorboard_dir="D:\\Code\\DeepLearning\\chatbot\\seq2seq_log")
    return model

  def train_model(self):
    train_xy, train_y = self.generate_trainig_data()
    model = self.seq2seq_model()
    model.fit(train_xy, train_y, n_epoch=1000, snapshot_epoch=False, batch_size=1)
    model.save(self.seq2seq_model_path)
    return model

  def load_model(self):
    model = self.seq2seq_model().load(self.seq2seq_model_path)
    return model

if __name__ == '__main__':
  word_vector_model_path = 'D:\\Code\\DeepLearning\\output\\mymodel'
  seq2seq_model_path = 'D:\\Code\\DeepLearning\\output\\seq2seq_model'
  input_file = 'D:\\Code\\DeepLearning\\output\\segment_pair.txt'
  my_seq2seq = Seq2Seq(word_vector_model_path, seq2seq_model_path, input_file)
  #my_seq2seq.train_model()
  model = my_seq2seq.load_model()
  trainXY, trainY = my_seq2seq.generate_trainig_data()
  predict = model.predict(trainXY)
  for sample in predict:
    print("predict answer")
