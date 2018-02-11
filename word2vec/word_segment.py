# -*- coding: utf-8 -*-
# @Time    : 2017/12/28 11:08
# @Author  : Alice
# @File    : word_segment.py
# @Desc    : 中文文档分词并去除停用词

import jieba
import os
import chardet

#创建停用词list
def stopwordlist(filepath):
  stopwords = [line.strip() for line in open(filepath,'r',encoding='utf-8').readlines()]
  return stopwords

def seg_sentence(input,output,stopwprds):
  input_file=open(input,'r')
  output_file=open(output,'a')
  while True:
    line = input_file.readline()
    if line:
      line = line.strip()
      seg_list = jieba.cut(line)
      segments = ''
      for word in seg_list:
        if word not in stopwprds:
          segments = segments + " " + word
      segments += '\n'
      output_file.write(segments)
    else:
      break
  input_file.close()
  output_file.close()

if __name__ == '__main__':
  segment_file = 'D:\\Code\\DeepLearning\\output\\segment.txt'
  stopwords = stopwordlist('D:\\Code\\DeepLearning\\chatbot\\word2vec\\stop_words.txt')
  for root, dirs, files in os.walk('D:\\Code\\DeepLearning\\output\\source'):
    for file in files:
      input = os.path.join(root,file)
      seg_sentence(input,segment_file,stopwords)