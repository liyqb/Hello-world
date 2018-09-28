#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
jobtype tags database machine-learning
liyq 2018.1.15
'''

from gensim.models import word2vec
import csv
import os
import codecs
import gensim
import jieba
import jieba.posseg as pseg
from gensim.models.word2vec import Word2Vec
import logging
import pandas as pd
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from datetime import datetime
import pyodbc
import re
import sys


def sql_jobname(var,from_table, keyword,nums):
    con = pyodbc.connect(SQL_par)
    con_s = con.cursor()
    if keyword!='':
        con_s.execute("select top "+str(nums)+" "+var+" from "+from_table+" where "+var+" like '%"+
                      keyword+"%'  group by "+var+" order by NEWID()")
    else:
        con_s.execute("select top " + str(nums) + " " + var + " from " + from_table +"  group by " +
                      var+" order by NEWID()")
    return con_s


def jobname_seg(con_s,update,filename):
    #print 'Step1: Read jobtype data and segment --------'+ datetime.now().strftime('%y-%m-%d %H:%M:%S')
    #jobtypes= []
    j_con1= open(filename,update)
    p = re.compile(ur'[\u4e00-\u9fa5]{1,}')
    for r in con_s:

        jieba.suggest_freq((u'软件', u'开发'), True)
        jieba.suggest_freq((u'软件', u'测试'), True)
        jieba.suggest_freq((u'前端', u'开发'), True)
        jieba.suggest_freq((u'数据库', u'管理员'), True)
        jieba.suggest_freq((u'电气', u'工程师'), True)
        jieba.suggest_freq((u'英语', u'老师'), True)
        jieba.suggest_freq((u'少儿', u'英语'), True)
        jieba.suggest_freq((u'文字', u'编辑'), True)
        if len(p.findall(r[0]))>0:
            rr= r[0]
            rr= rr.upper()
            rr= word_refine(rr)
            seg_list = jieba.cut(rr, HMM=True)
            #jobtypes.append(' '.join(seg_list))
            j_con1.write(' '.join(seg_list).encode('utf-8')+'\n')
    j_con1.close()


def word_refine(word):
    p = re.compile(r'<{1}\S*\W*\S*>')
    rr= word
    rr = re.sub(ur'\(', '<', rr)
    rr = re.sub(ur'（', '<', rr)
    rr = re.sub(ur'）', '>', rr)
    rr = re.sub(ur'\)', '>', rr)
    rr = re.sub(ur'【', '<', rr)
    rr = re.sub(ur'《', '<', rr)
    rr = re.sub(ur'】', '>', rr)
    rr = re.sub(ur'》', '>', rr)
    rr = p.sub('', rr)
    return rr


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r').readlines()]
    return stopwords


def movestopwords(sentence):
    stopwords = stopwordslist('jobtype_stop_words.txt')
    outstr = ''
    p= re.compile(r'[0-9]{2,}')
    p1= re.compile(r'[0-9]{1}[K]{1}')
    for word in sentence:
        if word not in stopwords:
            if word != '\t'and'\n':
                try:
                    if len(word.decode('utf-8'))>=2:
                        if p.match(word)==None:
                            if p1.match(word)==None:
                                outstr = outstr+' '+word+' '
                except UnicodeDecodeError as e:
                    print 'UnicodeDecodeError:'
    return outstr



def create_corpus(filename, corpusname,update):

    #print 'Step2: Move stop words and Create corpus data --------'+ datetime.now().strftime('%y-%m-%d %H:%M:%S')
    j_con3 = open(corpusname, update)
    j_con2= open(filename,"r")
    stopwords = stopwordslist('jobtype_stop_words.txt')
    jobtype_s= j_con2.readlines()
    jobtype_sub = []
    p = re.compile(r'[0-9]{2,}')
    for ws in jobtype_s:
        words= ws.split('\n')[0].split(' ')
        outstr=[]
        for word in words:
            if word not in stopwords:
                if (p.match(word)==None) :
                    try:
                        if len(word.decode('utf-8')) >= 2:
                            outstr.append(word)
                            jobtype_sub.append(word)
                    except UnicodeDecodeError as e:
                        print 'UnicodeDecodeError:'
        outstr= ' '.join(outstr)
        if outstr!='':
            j_con3.write(outstr.encode('utf-8')+'\n')
    j_con3.close()
    j_con2.close()



def count_word_freq(corpusname,freq):
    #print 'Step3: Count words Freq --------'+ datetime.now().strftime('%y-%m-%d %H:%M:%S')
    jobtype_sub= ' '.join(t.split('\n')[0] for t in open(corpusname,'r').readlines()).split(' ')
    j_dic= {}.fromkeys(list(set(jobtype_sub)),0)
    for t in jobtype_sub:
        j_dic[t]= j_dic.get(t,0)+1
    wordcount= pd.DataFrame({'word':j_dic.keys(),'freq':j_dic.values()})
    wordcount= wordcount.sort_values(by='freq',ascending=False)
    wordcount = wordcount[wordcount['freq'] >= freq]
    wordcount = wordcount[wordcount['word'] != '']
    wordcount.to_csv("jobtype-wordscount.csv", index=False, encoding='GB18030')
    print('       Complete!')


def word2vec_trainning(min_count,window,sg,update_tag,model_name,corpusname):
    #print 'Step4: Word2vec model training--------'+ datetime.now().strftime('%y-%m-%d %H:%M:%S')
    #time_str = datetime.now()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences= word2vec.LineSentence(corpusname)
    if sg==0:
        model_name= model_name+'0'
    if update_tag==0:
        model = Word2Vec(sentences, size=200, min_count= min_count, window=window, workers= 4,sg=sg,iter=20,hs=1, negative=0, seed= 12345)
        model.save(model_name)
    else:
        model = Word2Vec.load(model_name)
        model.build_vocab(sentences, update= True)
        model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
        model.save(model_name)


def model_test(model_name):
    model= Word2Vec.load(model_name)
    kws_t = pd.read_csv("jobtype_tags_database.csv", encoding='GB18030')
    kws= list(kws_t['kw'])
    lose_words=[]
    for w in kws:
        try:
            model.wv.word_vec(w)
        except BaseException as e:
            print e.message
            lose_words.append(w)
    return lose_words



def create_tags_basedata(model_name):
    model = Word2Vec.load(model_name)  # you can continue training with the loaded model!
    kws_t = pd.read_csv("jobtype_tags_database.csv", encoding='GB18030')

    j_con4 = open("jobtype-tags-similar-word2vec.csv", "wb")
    rew_kw_sim = csv.writer(j_con4)
    rew_kw_sim.writerow(['keyword', 'similar', 'value'])
    kws_t1 = list(kws_t['kw'])
    for i in range(0, 3):
        kws = kws_t[kws_t['class'] == 'pos' + str(5 - i)]
        if i == 1:
            kws = kws[(kws['flag'] == 'v') | (kws['flag'] == 'vn') | (kws['flag'] == 'j') ]
            kws = kws[kws['value'] >= 0.20]
            for w2, ww2 in kws.groupby('kw'):
                kws_t1.append(w2)
        if i == 2:
            kws = kws[(kws['flag'] == 'n') | (kws['flag'] == 'vn') | (kws['flag'] == 'j')\
                      | (kws['flag'] == 'x') | (kws['flag'] == 'l')]
            kws = kws[kws['value'] >= 0.20]
            for w2, ww2 in kws.groupby('kw'):
                kws_t1.append(w2)

        for w,ww in kws.groupby('kw'):

            try:
                y1= model.most_similar(w,topn=50)
            except KeyError:
                print 'KeyWarnings: '+w+'  not in vocabulary'
            else:
                #print '\n'+w1.encode('GB18030')
                for y2 in y1:
                    #print ' '*10+y2[0].encode('GB18030')+ '-'*5+str( y2[1])
                    rew_kw_sim.writerow([w.encode('GB18030'),y2[0].encode('GB18030'),y2[1]])

                    jieba.suggest_freq((u'软件', u'开发'), True)
                    jieba.suggest_freq((u'软件', u'测试'), True)
                    jieba.suggest_freq((u'前端', u'开发'), True)
                    jieba.suggest_freq((u'数据库', u'管理员'), True)
                    jieba.suggest_freq((u'电气', u'工程师'), True)
                    jieba.suggest_freq((u'英语', u'老师'), True)
                    jieba.suggest_freq((u'少儿', u'英语'), True)
                    jieba.suggest_freq((u'文字', u'编辑'), True)
                    #jieba.suggest_freq(y2[0], True)

                    for word, flag in pseg.cut(y2[0], HMM=True):
                        dup= [int(word== x) for x in kws_t1]
                        if sum(dup)== 0:
                            if len(word)>1:
                                kws_td= pd.DataFrame({'kw':word,'class':'pos'+str(4-i),'f_kw':w, 'flag':flag,'value':y2[1]}, index= [len(kws_t)+1])
                                kws_t= pd.concat([kws_t,kws_td])

    kws_t.to_csv("jobtype-tags-database-ML.csv", index=False)

    kws_t= pd.read_csv("jobtype-tags-database-ML.csv")
    tmp = kws_t[kws_t['class']=='pos1']

    kws_tt= pd.DataFrame({'pos1': [w for w in tmp['kw']]}, index=[x for x in range(len(tmp))])
    for i in range(4):
        t=[]
        tmp = kws_t[kws_t['class']=='pos'+str(i+2)]
        if i == 0:
            tmp = tmp[(tmp['flag'] == 'eng') | (tmp['flag'] == 'n') | (tmp['flag'] == 'nz') | (tmp['flag'] == 'i') | (tmp['flag'] == 'x') | (tmp['flag'] == 'l')]
            tmp = tmp[tmp['value'] >= 0.20]
            for w2, ww2 in tmp.groupby('kw'):
                t.append(w2)
        if i == 1:
            tmp = tmp[(tmp['flag'] == 'n') | (tmp['flag'] == 'vn') | (tmp['flag'] == 'j') | (tmp['flag'] == 'x') | (tmp['flag'] == 'l')]
            tmp = tmp[tmp['value'] >= 0.20]
            for w2, ww2 in tmp.groupby('kw'):
                t.append(w2)
        if i == 2:
            tmp = tmp[(tmp['flag'] == 'v') | (tmp['flag'] == 'vn') | (tmp['flag'] == 'j') ]
            tmp = tmp[tmp['value'] >= 0.20]
            for w2, ww2 in tmp.groupby('kw'):
                t.append(w2)
        if i == 3:
            for w2, ww2 in tmp.groupby('kw'):
                t.append(w2)
        kws_s= pd.DataFrame({'pos'+str(i+2): t})
        kws_tt= pd.concat([kws_tt,kws_s], axis=1, ignore_index=True)
    kws_tt.to_csv("jobtype-tags-database-ML-list.csv", index=False)

    kws_t= pd.read_csv("jobtype-tags-database-ML.csv")
    kws_tt= kws_t[kws_t['class']=='pos1']
    for i in range(4):
        tmp = kws_t[kws_t['class']=='pos'+str(i+2)]
        if i == 0:
            tmp = tmp[(tmp['flag'] == 'eng') | (tmp['flag'] == 'n') | (tmp['flag'] == 'nz') | (tmp['flag'] == 'i') | (tmp['flag'] == 'x')]
            tmp = tmp[tmp['value'] >= 0.20]

        if i == 1:
            tmp = tmp[(tmp['flag'] == 'n') | (tmp['flag'] == 'vn') | (tmp['flag'] == 'j') | (tmp['flag'] == 'x') | (tmp['flag'] == 'l')]
            tmp = tmp[tmp['value'] >= 0.20]

        if i == 2:
            tmp = tmp[(tmp['flag'] == 'v') | (tmp['flag'] == 'vn') | (tmp['flag'] == 'j') ]
            tmp = tmp[tmp['value'] >= 0.20]

        kws_tt= pd.concat([kws_tt,tmp], axis=0, ignore_index=True)
    kws_tt.to_csv("jobtype-tags-database-ML-sub.csv", index=False)
    kws_tt= kws_tt[['f_kw','kw','value']]
    kws_tt.to_csv("jobtype-tags-database-ML-sub2.csv", index=False)

    #word_vectors=KeyedVectors.load_word2vec_format('../R/result/word_vectors.txt', binary=False)




def create_tags_basedata_2(topn,freq):
    def get_tags_dic(jobtype_corpus, pos_kws):
        j_dic = []
        for i in range(len(pos_kws)): j_dic.append({})
        for ws in jobtype_corpus:
            j_cor = ws.decode('utf-8').split('\n')[0].split(' ')
            cor_kw_dup = list(set(j_cor).union(set(pos_kws)) ^ (set(pos_kws) ^ set(j_cor)))
            for w in cor_kw_dup:
                j_dic_i = [w == t for t in j_cor].index(True)
                kws_pos_i = [w == t for t in pos_kws].index(True)
                if j_dic_i > 0:
                    tmp = j_cor[j_dic_i - 1]
                    j_dic[kws_pos_i].update({tmp: j_dic[kws_pos_i].get(tmp, 0) + 1})
        return j_dic

    def get_pos_tags(tags_dic):
        p = re.compile(ur'[\u4e00-\u9fa5]{1,}')
        p1 = re.compile(r'[0-9]{1}[kKwWaA]{1}')
        p2= re.compile(r'\d{1,}')
        j_df_t = pd.DataFrame({'f_pos': [], 'pos': [], 'freq': [],'posseg':[]})
        for i in range(len(tags_database_dic[pos])):
            kw_p = []
            for d in tags_dic[i].keys():
                try:
                    kw_p.append(jieba_dic[d])
                except:
                    if (len(p.findall(d))==0) & (p1.match(d)==None):
                        kw_p.append('eng')
                    else:
                        kw_p.append('x')
            j_df = pd.DataFrame(
                {'f_pos': tags_database_dic[pos][i], 'pos': tags_dic[i].keys(), 'freq': tags_dic[i].values(), 'posseg': kw_p})
            j_df = j_df.sort_values(by='freq', ascending=False)
            j_df = j_df.iloc[0:topn, :]
            j_df_t = pd.concat([j_df_t, j_df], axis=0)
        j_df_t.to_csv('jobtype_tags_'+pos+'.csv', index=False, encoding='gb18030')
        return j_df_t

    def sel_tags_pos(var, tag, freq, pos):
        j_df_pos_s = j_df_t[j_df_t[var] == tag]
        j_df_pos_s = j_df_pos_s[j_df_pos_s['freq'] >= freq]
        tags_pos_s = list(j_df_pos_s['pos'])
        tags_database_dic[pos] = list(set(tags_database_dic[pos] + tags_pos_s))


    j_con1 = open('jobtype-corpus-sub.txt', 'r').readlines()
    jieba_dic={}
    for d in open('jobtype_dic.txt','r').readlines():
        ds= d.decode('utf-8').split('\n')[0].split()
        jieba_dic.update({ds[0]:ds[2]})
    print 'jieba dic OK!'

    tags_database_dic={}.fromkeys(['pos1','pos2','pos3','pos4','pos5'],[])
    tags_database_dic['pos5']= tags_database_dic['pos5']+kws_pos5
    tags_database_dic['pos1']= tags_database_dic['pos1']+kws_pos1
    tags_set= set(tags_database_dic['pos5']) | set(tags_database_dic['pos1'])
    for n in range(3):
        pos= 'pos'+str(5-n)
        s_pos= 'pos'+str(4-n)
        j_dic= get_tags_dic(j_con1,tags_database_dic[pos])
        j_df_t= get_pos_tags(j_dic)
        j_df_t= j_df_t.groupby(by=['pos','posseg'])['freq'].sum()
        j_df_t= j_df_t.to_dict()
        j_df_t= pd.DataFrame({'pos':[t[0] for t in j_df_t.keys()],
                              'posseg':[t[1] for t in j_df_t.keys()],
                              'freq':j_df_t.values()})
        j_df_t= j_df_t.sort_values(by='freq',ascending=False)
        j_df_t= j_df_t[j_df_t['freq']>=freq]
        if pos=='pos5':
            j_df_pos= j_df_t[(j_df_t['posseg']=='v') | (j_df_t['posseg']=='vn') |
                             (j_df_t['posseg']=='j') ]

            sel_tags_pos(var='posseg', tag='eng',freq=15,pos='pos2')
            sel_tags_pos(var='posseg',tag='nz', freq=5, pos='pos3')
        if pos=='pos4':
            j_df_pos = j_df_t[(j_df_t['posseg'] == 'n') | (j_df_t['posseg'] == 'vn') |
                              (j_df_t['posseg'] == 'j') | (j_df_t['posseg'] == 'l')]

            sel_tags_pos(var='posseg',tag='eng', freq=15, pos='pos2')
        if pos == 'pos3':
            j_df_pos = j_df_t[(j_df_t['posseg'] == 'n') |
                              (j_df_t['posseg'] == 'nz') | (j_df_t['posseg'] == 'i') |
                              (j_df_t['posseg'] == 'l')]
            j_df_pos_tmp= j_df_t[j_df_t['posseg'] == 'eng']
            j_df_pos= pd.concat([j_df_pos,j_df_pos_tmp[j_df_pos_tmp['freq']>=7]])

        tags_set= tags_set | set(tags_database_dic['pos2']) | set(tags_database_dic['pos3']) | set(tags_database_dic['pos4'])
        tags_pos_t= list(j_df_pos['pos'])
        tags_pos= list(set(tags_pos_t)-(tags_set & set(tags_pos_t)))
        #tags_set= tags_set | set(tags_pos)
        tags_database_dic[s_pos]= tags_database_dic[s_pos] +list(set(tags_pos))
        print s_pos,' tags OK!'
    tags_df= pd.concat([pd.DataFrame({'pos1':tags_database_dic['pos1']}),
                        pd.DataFrame({'pos2':sorted(tags_database_dic['pos2'])}),
                        pd.DataFrame({'pos3':tags_database_dic['pos3']}),
                        pd.DataFrame({'pos4':tags_database_dic['pos4']}),
                        pd.DataFrame({'pos5':tags_database_dic['pos5']})], axis=1)
    tags_df.to_csv('jobtype-tags-database-ML-list.csv', index=False, encoding='gb18030')



def create_tags_basedata_3(topn,freq):
    def get_tags_dic(jobtype_corpus, pos_kws):
        j_dic = []
        for i in range(len(pos_kws)): j_dic.append({})
        for ws in jobtype_corpus:
            j_cor = ws.decode('utf-8').split('\n')[0].split(' ')
            cor_kw_dup = list(set(j_cor).union(set(pos_kws)) ^ (set(pos_kws) ^ set(j_cor)))
            for w in cor_kw_dup:
                j_dic_i = [w == t for t in j_cor].index(True)
                kws_pos_i = [w == t for t in pos_kws].index(True)
                if j_dic_i > 0:
                    tmp = j_cor[j_dic_i - 1]
                    j_dic[kws_pos_i].update({tmp: j_dic[kws_pos_i].get(tmp, 0) + 1})
        return j_dic

    def get_pos_tags(tags_dic):
        p = re.compile(ur'[\u4e00-\u9fa5]{1,}')
        p1 = re.compile(r'[0-9]{1}[kKwWaA]{1}')
        p2= re.compile(r'\d{1,}')
        j_df_t = pd.DataFrame({'f_pos': [], 'pos': [], 'freq': [],'posseg':[],'p_value':[]})
        for i in range(len(tags_database_dic[pos])):
            kw_p = []
            freq_sum = sum(tags_dic[i].values())
            for d in tags_dic[i].keys():
                try:
                    kw_p.append(jieba_dic[d])
                except:
                    if (len(p.findall(d))==0) & (p1.match(d)==None):
                        kw_p.append('eng')
                    else:
                        kw_p.append('x')
            p_v= [float(t)/float(freq_sum) for t in tags_dic[i].values()]
            j_df = pd.DataFrame(
                {'f_pos': tags_database_dic[pos][i], 'pos': tags_dic[i].keys(),
                 'freq': tags_dic[i].values(), 'posseg': kw_p,'p_value':p_v})
            j_df = j_df.sort_values(by='freq', ascending=False)
            j_df = j_df.iloc[0:topn, :]
            j_df_t = pd.concat([j_df_t, j_df], axis=0)
        j_df_t = j_df_t[(j_df_t['freq'] >= freq) & (j_df_t['posseg'] != 'x') &
                        (j_df_t['posseg'] != 'nt') & (j_df_t['posseg'] != 'ns') &
                        (j_df_t['posseg'] != 'nrt') & (j_df_t['posseg'] != 'nr') &
                        (j_df_t['posseg'] != 't')]
        t_index= [t not in tags_database_dic['pos1'] for t in list(j_df_t['pos'])]
        j_df_t= j_df_t[t_index]
        j_df_t.to_csv('jobtype_tags_'+s_pos+'.csv', index=False, encoding='gb18030')
        return j_df_t


    j_con1 = open('jobtype-corpus-sub.txt', 'r').readlines()
    jieba_dic={}
    for d in open('jobtype_dic.txt','r').readlines():
        ds= d.decode('utf-8').split('\n')[0].split()
        jieba_dic.update({ds[0]:ds[2]})
    print 'jieba dic OK!'

    tags_database_dic={}.fromkeys(['pos1','pos2','pos3','pos4','pos5'],[])
    tags_database_dic['pos5']= tags_database_dic['pos5']+kws_pos5
    tags_database_dic['pos1']= tags_database_dic['pos1']+kws_pos1
    #tags_total=tags_database_dic['pos5']+tags_database_dic['pos1']
    for n in range(3):
        pos= 'pos'+str(5-n)
        s_pos= 'pos'+str(4-n)
        j_dic= get_tags_dic(j_con1,tags_database_dic[pos])
        j_df_t= get_pos_tags(j_dic)
        tags_pos= list(set(j_df_t['pos'])- set(tags_database_dic['pos5']))
        tags_database_dic[s_pos] = tags_database_dic[s_pos] + tags_pos
        print s_pos, ' tags OK!'
        #tags_total.extend(tags_pos)
    #tags_total= list(set(tags_total))
    #print 'Tags counts:', len(tags_total)
    tags_df = pd.concat([pd.DataFrame({'pos1': tags_database_dic['pos1']}),
                         pd.DataFrame({'pos2': sorted(tags_database_dic['pos2'])}),
                         pd.DataFrame({'pos3': tags_database_dic['pos3']}),
                         pd.DataFrame({'pos4': tags_database_dic['pos4']}),
                         pd.DataFrame({'pos5': tags_database_dic['pos5']})], axis=1)
    tags_df.to_csv('jobtype-tags-database-ML-list.csv', index=False, encoding='gb18030')
    print 'jobtype-tags-database-ML-list OK!'


def create_tags_basedata_4(topn,freq):
    def get_tags_dic(jobtype_corpus, pos_kws):
        j_dic = []
        for i in range(len(pos_kws)): j_dic.append({})
        for ws in jobtype_corpus:
            j_cor = ws.decode('utf-8').split('\n')[0].split(' ')
            cor_kw_dup = list(set(j_cor).union(set(pos_kws)) ^ (set(pos_kws) ^ set(j_cor)))
            for w in cor_kw_dup:
                j_dic_i = [w == t for t in j_cor].index(True)
                kws_pos_i = [w == t for t in pos_kws].index(True)
                if j_dic_i > 0:
                    tmp = j_cor[j_dic_i - 1]
                    j_dic[kws_pos_i].update({tmp: j_dic[kws_pos_i].get(tmp, 0) + 1})
        return j_dic

    def get_pos_tags(tags_dic):
        p = re.compile(ur'[\u4e00-\u9fa5]{1,}')
        p1 = re.compile(r'[0-9]{1}[kKwWaA]{1}')
        p2= re.compile(r'\d{1,}')
        j_df_t0 = pd.DataFrame({'f_pos': [], 'pos': [], 'freq': [], 'posseg': [], 'p_value_fpos': []})
        j_df_t = pd.DataFrame({'f_pos': [], 'pos': [], 'freq': [], 'posseg': [], 'p_value_fpos': [], 'p_value_pos': []})
        for i in range(len(tags_database_dic[pos])):
            kw_p = []
            freq_sum = sum(tags_dic[i].values())
            for d in tags_dic[i].keys():
                try:
                    kw_p.append(jieba_dic[d])
                except:
                    if (len(p.findall(d))==0) & (p1.match(d)==None):
                        kw_p.append('eng')
                    else:
                        kw_p.append('x')
            p_v= [float(t)/float(freq_sum) for t in tags_dic[i].values()]
            j_df = pd.DataFrame(
                {'f_pos': tags_database_dic[pos][i], 'pos': tags_dic[i].keys(),
                 'freq': tags_dic[i].values(), 'posseg': kw_p,'p_value_fpos':p_v})
            j_df = j_df.sort_values(by='freq', ascending=False)
            j_df = j_df.iloc[0:topn, :]
            j_df_t0 = pd.concat([j_df_t0, j_df], axis=0)
        j_df_t0 = j_df_t0[(j_df_t0['freq'] >= freq) & (j_df_t0['posseg'] != 'x') &
                          (j_df_t0['posseg'] != 'nt') & (j_df_t0['posseg'] != 'ns') &
                          (j_df_t0['posseg'] != 'nrt') & (j_df_t0['posseg'] != 'nr') &
                          (j_df_t0['posseg'] != 't') & (j_df_t0['posseg'] != 'ad') &
                          (j_df_t0['posseg'] != 'r')]
        s_pos_t = list(set(j_df_t0['pos']))
        for w in s_pos_t:
            j_df = j_df_t0[j_df_t0['pos'] == w].reset_index(drop=True)
            freq_sum = sum(j_df['freq'])
            pv = [float(t) / float(freq_sum) for t in list(j_df['freq'])]
            j_df = pd.concat([j_df, pd.DataFrame({'p_value_pos': pv})], axis=1)
            j_df_t = pd.concat([j_df_t, j_df])

        # t_index= [t not in tags_database_dic['pos1'] for t in list(j_df_t['pos'])]
        # j_df_t= j_df_t[t_index]
        j_df_t= j_df_t.reset_index(drop=True)
        p_value_tpos= [j_df_t['p_value_pos'][i]+j_df_t['p_value_fpos'][i] for i in range(len(j_df_t))]
        j_df_t= pd.concat([j_df_t, pd.DataFrame({'p_value_tpos':p_value_tpos})],axis=1)
        j_df_t = j_df_t.sort_values(by=['f_pos', 'freq'], ascending=False)
        j_df_t.to_csv('jobtype_tags_' + s_pos + '_M4.csv', index=False, encoding='gb18030')

        j_df = j_df_t.groupby(by='pos')['freq'].sum()
        j_df = j_df.to_dict()
        j_df = pd.DataFrame({'pos': [t for t in j_df.keys()], 'freq': j_df.values()})
        freq_sum = sum(j_df['freq'])
        pv = [float(t) / float(freq_sum) for t in list(j_df['freq'])]
        j_df = pd.concat([j_df, pd.DataFrame({'p_value': pv, 'pos_name': s_pos})], axis=1)

        return j_df_t,j_df


    j_con1 = open('jobtype-corpus-sub.txt', 'r').readlines()
    jieba_dic={}
    for d in open('jobtype_dic.txt','r').readlines():
        ds= d.decode('utf-8').split('\n')[0].split()
        jieba_dic.update({ds[0]:ds[2]})
    print 'jieba dic OK!'

    tags_database_dic={}.fromkeys(['pos1','pos2','pos3','pos4','pos5'],[])
    tags_database_dic['pos5']= tags_database_dic['pos5']+kws_pos5
    #tags_database_dic['pos1']= tags_database_dic['pos1']+kws_pos1
    tags_pos_pvalue = pd.DataFrame({'pos': [], 'freq': [], 'p_value': [], 'pos_name': []})
    for n in range(3):
        pos= 'pos'+str(5-n)
        s_pos= 'pos'+str(4-n)
        j_dic= get_tags_dic(j_con1,tags_database_dic[pos])
        j_df_t,j_df= get_pos_tags(j_dic)
        tags_pos_pvalue = pd.concat([tags_pos_pvalue, j_df])
        tags_pos= list(set(j_df_t['pos'])- set(tags_database_dic['pos5']))
        tags_database_dic[s_pos] = tags_database_dic[s_pos] + tags_pos
        print s_pos, ' tags OK!'

    tags_pos_pvalue = tags_pos_pvalue.reset_index(drop=True)
    tags_pos_pvalue.to_csv('tags_pos_PValue_M4.csv', index=False, encoding='gb18030')
    print ' tags PValue group by posn OK!'

    tags_df = pd.concat([pd.DataFrame({'pos1': kws_pos1}),
                         pd.DataFrame({'pos2': sorted(tags_database_dic['pos2'])}),
                         pd.DataFrame({'pos3': tags_database_dic['pos3']}),
                         pd.DataFrame({'pos4': tags_database_dic['pos4']}),
                         pd.DataFrame({'pos5': tags_database_dic['pos5']})], axis=1)
    tags_df.to_csv('jobtype-tags-database-ML-list-M4.csv', index=False, encoding='gb18030')
    print 'jobtype-tags-database-ML-list-M4 OK!'


def create_tags_basedata_5(topn,freq):
    def get_tags_dic(jobtype_corpus, pos_kws):
        j_dic = []
        for i in range(len(pos_kws)): j_dic.append({})
        for ws in jobtype_corpus:
            j_cor = ws.decode('utf-8').split('\n')[0].split(' ')
            cor_kw_dup = list(set(j_cor).union(set(pos_kws)) ^ (set(pos_kws) ^ set(j_cor)))
            for w in cor_kw_dup:
                j_dic_i = [w == t for t in j_cor].index(True)
                kws_pos_i = [w == t for t in pos_kws].index(True)
                if j_dic_i > 0:
                    tmp = j_cor[j_dic_i - 1]
                    j_dic[kws_pos_i].update({tmp: j_dic[kws_pos_i].get(tmp, 0) + 1})
        return j_dic

    def get_pos_tags(tags_dic):
        p = re.compile(ur'[\u4e00-\u9fa5]{1,}')
        p1 = re.compile(r'[0-9]{1}[kKwWaA]{1}')
        p2= re.compile(r'\d{1,}')
        j_df_t0 = pd.DataFrame({'f_pos': [], 'pos': [], 'freq': [],'posseg':[],'p_value_fpos':[]})
        j_df_t = pd.DataFrame({'f_pos': [], 'pos': [], 'freq': [], 'posseg': [], 'p_value_fpos': [],'p_value_pos':[]})
        for i in range(len(tags_database_dic[pos])):
            kw_p = []
            freq_sum = sum(tags_dic[i].values())
            for d in tags_dic[i].keys():
                try:
                    kw_p.append(jieba_dic[d])
                except:
                    if (len(p.findall(d))==0) & (p1.match(d)==None):
                        kw_p.append('eng')
                    else:
                        kw_p.append('x')
            p_v= [float(t)/float(freq_sum) for t in tags_dic[i].values()]
            j_df = pd.DataFrame(
                {'f_pos': tags_database_dic[pos][i], 'pos': tags_dic[i].keys(),
                 'freq': tags_dic[i].values(), 'posseg': kw_p,'p_value_fpos':p_v})
            j_df = j_df.sort_values(by='freq', ascending=False)
            j_df = j_df.iloc[0:topn, :]
            j_df_t0 = pd.concat([j_df_t0, j_df], axis=0)

        tags_dup= list(set(list(j_df_t0['pos']))-(set(kws_pos5) | set(kws_pos1)))
        t_index= [t in tags_dup for t in list(j_df_t0['pos'])]
        j_df_t0= j_df_t0[t_index]
        j_df_t0 = j_df_t0[(j_df_t0['freq'] >= freq) & (j_df_t0['posseg'] != 'x') &
                        (j_df_t0['posseg'] != 'nt') & (j_df_t0['posseg'] != 'ns') &
                        (j_df_t0['posseg'] != 'nrt') & (j_df_t0['posseg'] != 'nr') &
                        (j_df_t0['posseg'] != 't') & (j_df_t0['posseg'] != 'ad') &
                        (j_df_t0['posseg'] != 'r')]

        s_pos_t= list(set(j_df_t0['pos']))
        for w in s_pos_t:
            j_df= j_df_t0[j_df_t0['pos']==w].reset_index(drop=True)
            freq_sum= sum(j_df['freq'])
            pv= [float(t)/float(freq_sum) for t in list(j_df['freq'])]
            j_df= pd.concat([j_df,pd.DataFrame({'p_value_pos':pv})], axis=1)
            j_df_t= pd.concat([j_df_t,j_df])

        #t_index= [t not in tags_database_dic['pos1'] for t in list(j_df_t['pos'])]
        #j_df_t= j_df_t[t_index]

        j_df_t= j_df_t.sort_values(by=['f_pos','freq'], ascending=False)
        j_df_t.to_csv('jobtype_tags_'+s_pos+'_M5.csv', index=False, encoding='gb18030')

        j_df= j_df_t.groupby(by='pos')['freq'].sum()
        j_df = j_df.to_dict()
        j_df = pd.DataFrame({'pos': [t for t in j_df.keys()],'freq': j_df.values()})
        freq_sum= sum(j_df['freq'])
        pv= [float(t)/float(freq_sum) for t in list(j_df['freq'])]
        j_df= pd.concat([j_df, pd.DataFrame({'p_value':pv,'pos_name':s_pos})],axis=1)
        return j_df_t,j_df

    j_con1 = open('jobtype-corpus-sub.txt', 'r').readlines()
    jieba_dic={}
    for d in open('jobtype_dic.txt','r').readlines():
        ds= d.decode('utf-8').split('\n')[0].split()
        jieba_dic.update({ds[0]:ds[2]})
    print 'jieba dic OK!'

    tags_database_dic={}.fromkeys(['pos1','pos2','pos3','pos4','pos5'],[])
    tags_database_dic['pos5']= tags_database_dic['pos5']+kws_pos5
    #tags_database_dic['pos1']= tags_database_dic['pos1']+kws_pos1
    tags_pos_pvalue=pd.DataFrame({'pos':[],'freq':[],'p_value':[],'pos_name':[]})

    for n in range(3):
        pos= 'pos'+str(5-n)
        s_pos= 'pos'+str(4-n)
        j_dic= get_tags_dic(j_con1,tags_database_dic[pos])
        j_df_t,j_df= get_pos_tags(j_dic)
        tags_pos_pvalue = pd.concat([tags_pos_pvalue, j_df])
        tags_pos= list(set(j_df_t['pos']))
        tags_database_dic[s_pos] = tags_database_dic[s_pos] + tags_pos
        print s_pos, ' tags OK!'

    tags_pos_pvalue= tags_pos_pvalue.reset_index(drop=True)
    tags_pos_pvalue.to_csv('tags_pos_PValue_M5.csv', index=False, encoding='gb18030')
    print ' tags PValue group by posn OK!'

    tags_pos_pvalue= tags_pos_pvalue.groupby(by='pos')
    tags_dic={}.fromkeys(['pos2','pos3','pos4'],[])
    for w,rs in tags_pos_pvalue:
        p_max= rs['pos_name'][rs[rs['p_value']==max(rs['p_value'])].index[0]]
        tags_dic[p_max]= tags_dic[p_max]+[w]


    tags_df = pd.concat([pd.DataFrame({'pos1': kws_pos1}),
                         pd.DataFrame({'pos2': tags_dic['pos2']}),
                         pd.DataFrame({'pos3': tags_dic['pos3']}),
                         pd.DataFrame({'pos4': tags_dic['pos4']}),
                         pd.DataFrame({'pos5': tags_database_dic['pos5']})], axis=1)
    tags_df.to_csv('jobtype-tags-database-ML-list-M5.csv', index=False, encoding='gb18030')
    print 'jobtype-tags-database-ML-list-M5 OK!'



def create_tags_mat(pvalue_mode,tags_mode):
    tags_df= pd.read_csv('jobtype-tags-database-ML-list-'+tags_mode+'.csv', encoding='gb18030')
    tags= []
    for i in range(5):
        tags.extend(list(tags_df['pos'+str(i+1)]))
    tags= list(set(tags))[1:]
    tags_df= pd.DataFrame({'tags':tags,'index':range(len(tags))})
    tags_index=tags_df.set_index('tags').T.to_dict('list')
    tags_mat= np.zeros([len(tags),len(tags)])
    for i in range(3):
        tags_df= pd.read_csv('jobtype_tags_pos'+str(4-i)+'_'+tags_mode+'.csv', encoding='gb18030')
        for j in range(tags_df.shape[0]):
            x= tags_index[tags_df['pos'][j]]
            y= tags_index[tags_df['f_pos'][j]]
            tags_mat[x,y]= tags_df[pvalue_mode][j]
    tags_df= pd.DataFrame(data=tags_mat,index=tags,columns=tags)
    tags_df.to_csv('jobtype_tags_matrix_'+tags_mode+'.csv', encoding='gb18030')
    return tags_index,tags_mat




if __name__ == '__main__':

    default_encoding = 'utf-8'
    if sys.getdefaultencoding() != default_encoding:
        reload(sys)
        sys.setdefaultencoding(default_encoding)
    # print sys.getdefaultencoding()

    SQL_con = ["DSN=yxxy-sql;UID=sa;PWD=yxxy",
               "DSN=jobdata;UID=sa;PWD=0829",
               "DSN=SQL_server;UID=sa;PWD=yxxy",
               "DSN=SQL;UID=sa;PWD=yxxy"]

    SQL_par = SQL_con[1]

    jieba.load_userdict('jobtype_dic.txt')
    indus = ['Bdata', 'Bintel', 'Beco', 'Bcul', 'Bhealth', 'Bservice']
    kws_base = pd.read_csv('jobtype_tags_database.csv', encoding='gb18030')
    kws_pos5= kws_base[kws_base['class']=='pos5']
    kws_pos5= list(kws_pos5['kw'])
    kws_pos1= kws_base[kws_base['class']=='pos1']
    kws_pos1= list(kws_pos1['kw'])

    '''
    print 'Step1: Read jobtype data and segment --------' + datetime.now().strftime('%y-%m-%d %H:%M:%S')
    ####  1
    con_s= sql_jobname(var='jobname',from_table='joblist',keyword='',nums=400000)
    jobname_seg(con_s= con_s,update='w',filename="jobtype-corpus-add.txt")
    
    ####  2
    for w in kws_pos5:
        con_s = sql_jobname(var='jobName', from_table='joblist_2', keyword=w, nums=200000)
        jobname_seg(con_s=con_s, update='a', filename="jobtype-corpus.txt")
        print w, ' jobname seg ok!'
    con_s = sql_jobname(var='jobname', from_table='joblist', keyword='', nums=200000)
    jobname_seg(con_s=con_s, update='a', filename="jobtype-corpus.txt")
    
    print 'Step2: Move stop words and Create corpus data --------' + datetime.now().strftime('%y-%m-%d %H:%M:%S')
    create_corpus(filename="jobtype-corpus-add.txt", corpusname="jobtype-corpus-sub.txt", update='a')
    
    print 'Step3: Count words Freq --------' + datetime.now().strftime('%y-%m-%d %H:%M:%S')
    count_word_freq(corpusname='jobtype-corpus-sub.txt', freq=5)
    
    print 'Step4: Word2vec model training--------' + datetime.now().strftime('%y-%m-%d %H:%M:%S')
    word2vec_trainning(min_count=2, window=4, sg=0 ,update_tag=0, model_name='jobtype_model',
                       corpusname='jobtype-corpus-sub.txt')

    
    print 'Step5: Test Word2vec model--------' + datetime.now().strftime('%y-%m-%d %H:%M:%S')
    lose_words = model_test('jobtype_model')
    #lose_words = model_test('jobtype_model0')
    
    if len(lose_words) > 0:
        print len(lose_words), ' words not in vocabulary! Try to update model!'
        for w in lose_words:
            con_s = sql_jobname(var='jobname', from_table='joblist', keyword=w, nums=1000)
            jobname_seg(con_s=con_s, update='w', filename="jobtype-corpus-add1.txt")
        create_corpus(filename="jobtype-corpus-add1.txt", corpusname="jobtype-corpus-add-sub.txt", update='w')
        #count_word_freq(corpusname='jobtype-corpus-sub.txt', freq=2)
        word2vec_trainning(min_count=2, window=4, sg=0,update_tag=1, model_name='jobtype_model',
                           corpusname='jobtype-corpus-add-sub.txt')
        print 'Update model OK!'
        print 'Retest Word2vec model--------' + datetime.now().strftime('%y-%m-%d %H:%M:%S')
        lose_words = model_test('jobtype_model')
        if len(lose_words) > 0:
            print len(lose_words), ' words not in vocabulary! Please check model!'
        else:
            print 'Test model OK!'
    else:
        print 'Test model OK!'
    '''
    print 'Step6: Create jobtype tags basedata--------' + datetime.now().strftime('%y-%m-%d %H:%M:%S')
    '''
    ### Concat jotype corpus
    j_con1= open('jobtype-corpus-sub.txt','a')
    j_con2= open('jobtype-corpus-add-sub.txt','r')
    for j in j_con2.readlines(): 
        j_con1.writelines(j)
    j_con1.close()
    j_con2.close()
    '''
    
    # create_tags_basedata(model_name='jobtype_model')
    #create_tags_basedata_2(topn=50, freq=20)
    #create_tags_basedata_3(topn=80, freq=15)

    create_tags_basedata_4(topn=1000, freq=25)
    #create_tags_basedata_5(topn=1000, freq=30)

    print 'Step7: Create jobtype tags matrix --------'+ datetime.now().strftime('%y-%m-%d %H:%M:%S')
    ## pvalue_mode='p_value_pos': group by pos为基数计算f_pos的p_value
    ## pvalue_mode='p_value_fpos': group by fpos为基数计算pos的p_value
    create_tags_mat(pvalue_mode='p_value_tpos',tags_mode='M4')
    #create_tags_mat(pvalue_mode='p_value_pos', tags_mode='M5')

    print 'Complete --------' + datetime.now().strftime('%y-%m-%d %H:%M:%S')



