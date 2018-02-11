# -*- coding: utf-8 -*-
# @Time    : 2017/12/28 11:08
# @Author  : Alice
# @File    : word_segment.py
# @Desc    : 使用tensorflow输出词向量

from gensim.models import word2vec
import multiprocessing
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def build_word2vec_model(source,modelpath):
  """
  生成word2vec模型
  :param source: 文件路径
  :param modelpath: 模型保存路径
  :return:
  """
  sentences = word2vec.LineSentence(source)
  model = word2vec.Word2Vec(sentences, min_count=8,size=200,workers=multiprocessing.cpu_count())
  model.save(modelpath)
  #model.save_word2vec_format( 'D:\\Code\\DeepLearning\\output\\mymodel.bin')

def build_word_dict(model_path):
  """
  获取word2vec模型的所有词向量
  :param model_path: 已经训练好的word2vec模型保存路径
  :return:
  """
  model = word2vec.Word2Vec.load(model_path)
  vocab = model.wv.vocab
  word_vector = {}
  for word in vocab:
    word_vector[word] = model[word]
  return word_vector

if __name__ == '__main__':
  source = 'D:\\Code\\DeepLearning\\output\\segment.txt'
  modelpath = 'D:\\Code\\DeepLearning\\output\\mymodel'
  #build_word2vec_model(source,modelpath)

  word_vector_dict = build_word_dict(modelpath)
  if '前' in word_vector_dict:
    print('有')
  else:
    print('没有')

