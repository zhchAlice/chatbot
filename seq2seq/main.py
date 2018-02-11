# -*- coding: utf-8 -*-
# @Time    : 2018/2/8 17:07
# @Author  : Alice
# @File    : main.py
# @Desc    :
file_object = open('D:\\Code\\DeepLearning\\output\\segment_pair.txt','r',encoding='utf-8')
while True:
  line = file_object.readline()
  if line:
    line_pair = line.split("|")
    line_question_words = line_pair[0].split(" ")
    line_answer_words = line_pair[1].split(" ")