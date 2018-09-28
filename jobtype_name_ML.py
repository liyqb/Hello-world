#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
jobtype name database machine-learning
liyq 2018.1.22
'''


import pandas as pd
import time
import sys
import jieba
import re
import pyodbc
from gensim.models.word2vec import Word2Vec
import numpy as np

default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
    reload(sys)
    sys.setdefaultencoding(default_encoding)
    # print sys.getdefaultencoding()


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


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r').readlines()]
    return stopwords

####
def create_jobtype(jobname_data):
    #print 'Step1: Read test jobname data, create jobtype tags and jobtype name'

    kws_base= pd.read_csv("jobtype-tags-database-ML-list.csv", encoding='GB18030')
    s1,s2,s3,s4=[],[],[],[]
    word_freq= pd.read_csv('jobtype-wordscount.csv',encoding='gb18030')
    word_dic= word_freq.set_index('word').T.to_dict('list')
    p1 = re.compile(ur'[\u4e00-\u9fa5]{1,}')
    for r in jobname_data:
        if len(p1.findall(r[0]))>0:
            s= r[0]
            #s= word_refine(s)
            s= s.upper()
            jieba.suggest_freq((u'软件', u'开发'), True)
            jieba.suggest_freq((u'软件', u'测试'), True)
            jieba.suggest_freq((u'前端', u'开发'), True)
            jieba.suggest_freq((u'数据库', u'管理员'), True)
            jieba.suggest_freq((u'电气', u'工程师'), True)
            jieba.suggest_freq((u'英语', u'老师'), True)
            jieba.suggest_freq((u'少儿', u'英语'), True)
            jieba.suggest_freq((u'文字', u'编辑'), True)
            '''
            ### 按照TF-IDF提取jobname的关键词和权重，遇到相同pos的词取权重大的词为此pos词
            s= ' '.join(jieba.cut(s,HMM=True))
            for x,w in ja.extract_tags(s, withWeight=True):
                if x not in stopwords:
                    s1.append(r[0])
                    s2.append(x)
                    s3.append(w)
                    yy = ''
                    x= x.upper()
                    for j in range(5):
                        dup=[int(x==y) for y  in kws_base[str(j)]]
                        if sum(dup) > 0:
                            yy='pos'+str(j+1)
                    s4.append(yy)
            '''
            ### 用jobtype-wordscount.csv表中的freq大小作为词的权重，遇到相同pos的词取权重大的词为此pos词
            for x in jieba.cut(s.strip(),HMM=True):
                s1.append(r[0])
                s2.append(x)
                try:
                    s3.append(word_dic[x][0])
                except KeyError:
                    s3.append(0)
                yy = ''
                x = x.upper()
                for j in range(5):
                    dup = [int(x == y) for y in kws_base[str(j)]]
                    if sum(dup) > 0:
                        yy = 'pos' + str(j + 1)
                s4.append(yy)

    test1= pd.DataFrame({'jobname':s1,'kws':s2,'weight':s3,'pos':s4})
    test1.to_csv("jobname_kws.csv",index=False, encoding='GB18030')

    s1,s2,p1,p2,p3,p4,p5,p45=[],[],[],[],[],[],[],[]
    for x,w in test1.groupby('jobname'):
        s1.append(x)
        w1= w[w['pos']!='']
        w1= w1.sort_values(by=['pos','weight'],ascending= False)
        w1= w1.drop_duplicates(['pos'])
        w1 = w1.sort_values(by=['pos'], ascending= True)
        w1=pd.DataFrame.reset_index(w1,drop=True)
        x1= list(w1['kws'])
        x2=''.join(x1)
        s2.append(x2)
        p1.append('none')
        p2.append('none')
        p3.append('none')
        p4.append('none')
        p5.append('none')
        for i in range(len(w1)):
            if w1['pos'][i]=='pos1':
                p1[-1]= w1['kws'][i]
            if w1['pos'][i]=='pos2':
                p2[-1]=w1['kws'][i]
            if w1['pos'][i]=='pos3':
                p3[-1]=w1['kws'][i]
            if w1['pos'][i]=='pos4':
                p4[-1]=w1['kws'][i]
            if w1['pos'][i]=='pos5':
                p5[-1]=w1['kws'][i]
        p45.append(p4[-1]+p5[-1])
        if p2[-1]!='none':
            p2[-1]=p2[-1].upper()

    test2= pd.DataFrame({'jobname':s1,'jobtype':s2,'pos1':p1,'pos2':p2,'pos3':p3,'pos4':p4,'pos5':p5,'pos_p':p45})
    test_t= pd.read_csv("jobname_jobtype.csv", encoding='gb18030')
    test_t= pd.concat([test_t, test2])
    #test_t = test_t.loc[test_t.duplicated('jobname')== False,:]
    test_t.to_csv("jobname_jobtype.csv",index=False,encoding='GB18030')


def create_jobtype2(jobname_data,freq):
    #print 'Step1: Read test jobname data, create jobtype tags and jobtype name'

    def get_pos(f_pos_list,pos_list, tags_pos):
        t_fp,t_p,t_freq,fp_freq,p_freq=[],[],[],[],[]
        for n in f_pos_list:
            for m in pos_list:
                t_fp.append(n)
                t_p.append(m)
                try: fp_freq.append(word_dic[n][0])
                except: fp_freq.append(0)
                try: p_freq.append(word_dic[m][0])
                except: p_freq.append(0)
                try:
                    t_freq.append(tags_pos[(tags_pos['f_pos'] == n) & (tags_pos['pos'] == m)]['freq'][0])
                except IndexError:
                    t_freq.append(0)
        t_df = pd.DataFrame({'f_pos': t_fp,'fp_freq':fp_freq,'pos':t_p,'p_freq':p_freq,'freq':t_freq})
        if max(t_df['freq'])>=freq:
            f_pos= t_df[t_df['freq']==max(t_df['freq'])]['f_pos'].values[0]
            pos= t_df[t_df['freq']==max(t_df['freq'])]['pos'].values[0]
        else:
            f_pos= t_df[t_df['fp_freq']==max(t_df['fp_freq'])]['f_pos'].values[0]
            pos='none'
        return f_pos,pos


    kws_base= pd.read_csv("jobtype-tags-database-ML-list.csv", encoding='GB18030')
    tags_pos5= pd.read_csv('jobtype_tags_pos5.csv', encoding='gb18030')
    tags_pos4 = pd.read_csv('jobtype_tags_pos4.csv', encoding='gb18030')
    tags_pos3 = pd.read_csv('jobtype_tags_pos3.csv', encoding='gb18030')
    jn,jt,p1,p2,p3,p4,p5,p45=[],[],[],[],[],[],[],[]
    word_freq= pd.read_csv('jobtype-wordscount.csv',encoding='gb18030')
    word_dic= word_freq.set_index('word').T.to_dict('list')
    pr = re.compile(ur'[\u4e00-\u9fa5]{1,}')
    for r in jobname_data:
        if len(pr.findall(r[0]))>0:
            jn.append(r[0])
            s= r[0]
            #s= word_refine(s)
            s= s.upper()
            jieba.suggest_freq((u'软件', u'开发'), True)
            jieba.suggest_freq((u'软件', u'测试'), True)
            jieba.suggest_freq((u'前端', u'开发'), True)
            jieba.suggest_freq((u'数据库', u'管理员'), True)
            jieba.suggest_freq((u'电气', u'工程师'), True)
            jieba.suggest_freq((u'英语', u'老师'), True)
            jieba.suggest_freq((u'少儿', u'英语'), True)
            jieba.suggest_freq((u'文字', u'编辑'), True)
            ### 用jobtype-wordscount.csv表中的freq大小作为词的权重，遇到相同pos的词取权重大的词为此pos词
            ws= [x for x in jieba.cut(s.strip(), HMM=True)]
            t5= list(set(ws) & set(kws_base['pos5']))
            t4= list(set(ws) & set(kws_base['pos4']))
            t3 = list(set(ws) & set(kws_base['pos3']))
            t2 = list(set(ws) & set(kws_base['pos2']))
            t1 = list(set(ws) & set(kws_base['pos1']))
            t45_p4,t45_p5,t45_freq=[],[],[]
            p1.append('')
            p2.append('')
            p3.append('')
            p4.append('')
            p5.append('')

            if len(t5)>0:
                if len(t4)>0:
                    t_fp,t_p= get_pos(f_pos_list=t5,pos_list=t4,tags_pos=tags_pos5)
                    p5[-1]=t_fp
                    p4[-1]=t_p
                else:
                    p4[-1]='none'
                    if len(t3)>0:
                        t_fp, t_p = get_pos(f_pos_list=t5, pos_list=t3, tags_pos=tags_pos5)
                        p5[-1]= t_fp

            else:  p5.append('none')
            if t4[-1] != 'none':
                if len(t3) > 0:
                    t_fp, t_p = get_pos(f_pos_list=[t4[-1]], pos_list=t3,tags_pos=tags_pos4)
                    p3.append(t_p)
                else: p3.append('none')
            else:
                if len(t3) > 0:
                    t_fp, t_p = get_pos(f_pos_list=[t5[-1]], pos_list=t3,tags_pos=tags_pos5)
                    p3.append(t_p)
                else: p3.append('none')
            if t3[-1] != 'none':
                if len(t2) > 0:
                    t_fp, t_p = get_pos(f_pos_list=[t3[-1]], pos_list=t2,tags_pos=tags_pos3)
                    p2.append(t_p)
                else:
                    p2.append('none')
            else:
                if len(t2) > 0:
                    t_fp, t_p = get_pos(f_pos_list=[t4[-1]], pos_list=t2,tags_pos=tags_pos4)
                    if t_p == 'none':
                        t_fp, t_p = get_pos(f_pos_list=[t5[-1]], pos_list=t2,tags_pos=tags_pos5)
                        p2.append(t_p)
                    else:
                        p2.append(t_p)
                else:
                    p2.append('none')
            if len(t3) > 0:
                t_fp, t_p = get_pos(f_pos_list=[t5[-1]], pos_list=t3,tags_pos=tags_pos5)
                p3.append(t_p)
            else:
                p3.append('none')
            if p3[-1] != 'none':
                if len(t2) > 0:
                    t_fp, t_p = get_pos(f_pos_list=[t3[-1]], pos_list=t2,tags_pos=tags_pos3)
                    p2.append(t_p)
                else:
                    p2.append('none')
            else:
                if len(t2) > 0:
                    t_fp, t_p = get_pos(f_pos_list=[t4[-1]], pos_list=t2,tags_pos=tags_pos4)
                    if t_p == 'none':
                        t_fp, t_p = get_pos(f_pos_list=[t5[-1]], pos_list=t2,tags_pos=tags_pos5)
                        p2.append(t_p)
                    else:
                        p2.append(t_p)
                else:
                    p2.append('none')



    test2= pd.DataFrame({'jobname':jn,'jobtype':jt,'pos1':p1,'pos2':p2,'pos3':p3,'pos4':p4,'pos5':p5,'pos_p':p45})
    test_t= pd.read_csv("jobname_jobtype.csv", encoding='gb18030')
    test_t= pd.concat([test_t, test2])
    #test_t = test_t.loc[test_t.duplicated('jobname')== False,:]
    test_t.to_csv("jobname_jobtype.csv",index=False,encoding='GB18030')


def create_jobtype3(jobname_data,pvalue_mode,tags_mode):

    def get_max_p_jobtype():
        tags_links, p_value,tags_links_sub=[],[],[]
        ts=[]
        for t1 in tps_dic['pos1']:
            ts.append(t1)
            for t2 in tps_dic['pos2']:
                ts.append(t2)
                for t3 in tps_dic['pos3']:
                    ts.append(t3)
                    for t4 in tps_dic['pos4']:
                        ts.append(t4)
                        for t5 in tps_dic['pos5']:
                            ts.append(t5)
                            tags_links.append(ts)
                            ts= ts[0:-1]
                        ts = ts[0:-1]
                    ts = ts[0:-1]
                ts = ts[0:-1]
            ts = ts[0:-1]

        for tl in tags_links:
            tl_sub= []
            for w in tl:
                if (w not in tl_sub) & (w !='none'):
                    tl_sub.append(w)
            #tl_sub= tl
            pv=0.
            i=0
            if len(tl_sub)>1:
                while i+1<len(tl_sub):
                    x= tags_index[tl_sub[i]]
                    y= tags_index[tl_sub[i+1]]
                    if tl_sub[i] in list(kws_base['pos1']):
                        tags_mat[x,y]=0
                        pv = pv + tags_mat[x, y]
                    else:
                        if tags_mat[x,y]>0 :
                            pv= pv+ tags_mat[x,y]
                        else:
                            pv=[0.]
                            break
                    i+=1
                #p_value.append(pv[0]/(len(tl_sub)-1))
                p_value.append(pv[0])
                tags_links_sub.append(tl_sub)
            else:
                p_value.append(0.)
                tags_links_sub.append(tl_sub)

        tags_links= ['-'.join(t) for t in tags_links]
        tags_links_sub= ['-'.join(t) for t in tags_links_sub]
        return tags_links,p_value,tags_links_sub


    # print 'Step1: Read test jobname data, create jobtype tags and jobtype name'
    kws_base = pd.read_csv("jobtype-tags-database-ML-list-"+tags_mode+".csv", encoding='GB18030')
    import jobtype_tags_ML as jtm
    tags_index, tags_mat=jtm.create_tags_mat(pvalue_mode=pvalue_mode,tags_mode=tags_mode)
    print 'Jobtype tags matrix OK!'
    jobtype_df_t= pd.DataFrame()
    jobtype_df= pd.DataFrame()
    jn = []
    pr = re.compile(ur'[\u4e00-\u9fa5]{1,}')
    for r in jobname_data:
        if len(pr.findall(r[0])) > 0:
            tps=[]
            tps_dic={}.fromkeys(['pos1','pos2','pos3','pos4','pos5'],[])
            jn.append(r[0])
            s = r[0]
            s= word_refine(s)
            s = s.upper()
            jieba.suggest_freq((u'软件', u'开发'), True)
            jieba.suggest_freq((u'软件', u'测试'), True)
            jieba.suggest_freq((u'前端', u'开发'), True)
            jieba.suggest_freq((u'数据库', u'管理员'), True)
            jieba.suggest_freq((u'电气', u'工程师'), True)
            jieba.suggest_freq((u'英语', u'老师'), True)
            jieba.suggest_freq((u'少儿', u'英语'), True)
            jieba.suggest_freq((u'文字', u'编辑'), True)
            ws = [x for x in jieba.cut(s.strip(), HMM=True)]
            for i in range(5):
                if i==0:
                    tp = list(set(ws) & set(kws_base['pos'+str(i+1)]))
                else:
                    tp = list(set(ws) & (set(kws_base['pos' + str(i + 1)]) - set(kws_base['pos1'])))
                if len(tp)>0:
                    tps.append(tp)
                    tps_dic['pos'+str(i+1)]= tps_dic['pos'+str(i+1)]+tp
                else:
                    tps_dic['pos' + str(i + 1)] = tps_dic['pos' + str(i + 1)] + ['none']

            jobtypes,p_values,jobtypes_sub= get_max_p_jobtype()
            j_df= pd.DataFrame({'jobname':r[0],'jobtype_tags':jobtypes,'jobtype':jobtypes_sub,'p_value':p_values})
            j_df=j_df.sort_values(by='p_value', ascending=False)
            j_df_max= j_df[j_df['p_value']==max(j_df['p_value'])]
            if j_df_max.shape[0]>1:
                j_len=[len(t.split('-')) for t in j_df_max['jobtype']]
                j_len= [t==max(j_len) for t in j_len]
                j_df_max= j_df_max[j_len]
                j_df_max= j_df_max.iloc[0:1,:]
            jobtype_df= pd.concat([jobtype_df,j_df_max])
            jobtype_df_t= pd.concat([jobtype_df_t,j_df])

    jobtype_df_t.to_csv("jobname_jobtype_by_pv_"+tags_mode+"_"+pvalue_mode+".csv", index=False, encoding='GB18030')
    jobtype_df.to_csv('jobname_jobtype_by_max_pv_'+tags_mode+"_"+pvalue_mode+'.csv', index=False, encoding='gb18030')



def jobtype_refine2(jobtype_df, tags_mode,pvalue_mode,pos_w1,pos_w2,p_min):
    jobtype_df= pd.read_csv(jobtype_df, encoding='gb18030')
    kws_base = pd.read_csv("jobtype-tags-database-ML-list-" + tags_mode + ".csv", encoding='GB18030')
    tags_base = pd.read_csv('jobtype_tags_database.csv', encoding='gb18030')
    kws_pos5m= list(tags_base[tags_base['type']=='m']['kw'])
    tags_pos_pvalue = pd.read_csv('tags_pos_PValue_' + tags_mode + '.csv', encoding='gb18030')
    jobtype_re = list(jobtype_df['jobtype_tags'])
    jobtype_re_pmax,jobtype_re2 = [],[]
    tags_pos2 = pd.read_csv('jobtype_tags_pos2_' + tags_mode + '.csv', encoding='gb18030')
    tags_pos3 = pd.read_csv('jobtype_tags_pos3_' + tags_mode + '.csv', encoding='gb18030')
    tags_pos4 = pd.read_csv('jobtype_tags_pos4_' + tags_mode + '.csv', encoding='gb18030')
    tags_pos=[]
    for i in range(len(jobtype_re)):
        ## 定位tag的pos位置
        tags_t = jobtype_re[i].split('-')
        for m in range(1,4):
            if tags_t[m] in list(kws_base['pos1']):
                tags_t[m]='none'
        tags = tags_t[1:4]
        t_df = pd.DataFrame({'tags': tags, 'pos': ['pos2', 'pos3', 'pos4']})
        t_df_gr = t_df.groupby(by='tags')
        for w, rs in t_df_gr:
            if (len(rs) > 1) & (w !='none'):
                tp_df = tags_pos_pvalue[tags_pos_pvalue['pos'] == w]
                tp_index= [tp_df.loc[tp_df.index[n],'pos_name'] in list(rs['pos']) for n in range(len(tp_df))]
                tp_df= tp_df[tp_index]
                try:
                    p_max = list(tp_df[tp_df['p_value'] != max(tp_df['p_value'])]['pos_name'])
                except BaseException as e :
                    print e.message
                    print w, rs
                else:
                    for m in p_max:
                        t_df.loc[t_df[t_df['pos'] == m].index[0], 'tags'] = 'none'
                    tags_t[1:4] = list(t_df['tags'])
                    break
        jobtype_re[i] = '-'.join(tags_t)

        ## 根据最大概率路径添补部分为‘none’的tag
        def get_tag_101(tags_posn1, tags_posn2, posn1, posn2, w_posn1,w_posn2):
            p_max = 0.
            w_max = 'none'
            t_df = tags_posn1.loc[tags_posn1['pos'] == w_posn1, ['f_pos', pvalue_mode]]
            t_df = t_df[t_df[pvalue_mode] >= p_min]
            if len(t_df) > 0:
                dic = t_df.set_index('f_pos').T.to_dict('list')
                t_df = tags_posn2.loc[tags_posn2['f_pos'] == w_posn2,['pos',pvalue_mode]]
                dic1 = t_df.set_index('pos').T.to_dict('list')
                ws = list((set(dic.keys()) & set(dic1.keys())) - set(kws_base['pos1']))
                for w in ws:
                    p = dic[w][0]*pos_w1 + dic1[w][0]*pos_w2
                    if p > p_max:
                        p_max = p
                        w_max = w
                if p_max<=p_min:
                    p_max=0.
                    w_max='none'
            return w_max,p_max

        def get_tag_1001(tags_posn1, tags_posn2, tags_posn3,posn1, posn2):
            p_max_t = 0.
            w1_max, w2_max = 'none', 'none'
            t_df = tags_posn1.loc[tags_posn1['pos'] == t_dic[posn1], ['f_pos', pvalue_mode]]
            t_df = t_df[t_df[pvalue_mode] >= p_min]
            if len(t_df) > 0:
                dic = t_df.set_index('f_pos').T.to_dict('list')
                for w in list(set(dic.keys())-set(kws_base['pos1'])):
                    w_pos4,p_pos4= get_tag_101(tags_posn2,tags_posn3,'pos3',posn2,w,t_dic[posn2])
                    if w_pos4!='none':
                        p_t= dic[w][0]*pos_w1+p_pos4*pos_w2
                        if p_t>p_max_t:
                            p_max_t= p_t
                            w1_max=w
                            w2_max= w_pos4
                if p_max_t<=p_min:
                    p_max_t=0.
                    w1_max,w2_max='none','none'
            return w1_max,w2_max,p_max_t

        def get_tag_1100(tags_posn1, tags_posn2,posn1):
            p_max_t, p_t = 0., 0.
            w1_max, w2_max, w_pos5 = 'none', 'none', 'none'
            t_df = tags_posn1.loc[tags_posn1['pos'] == t_dic[posn1], ['f_pos', pvalue_mode]]
            t_df= t_df[t_df[pvalue_mode]>=p_min]
            if len(t_df)>0:
                dic = t_df.set_index('f_pos').T.to_dict('list')
                for w in list(set(dic.keys())-set(kws_base['pos1'])):
                    t_df= tags_posn2.loc[tags_posn2['pos'] == w, ['f_pos', pvalue_mode]]
                    t_df= t_df[t_df[pvalue_mode]>=p_min]
                    if len(t_df)>0:
                        t_index = [ww not in kws_pos5m for ww in list(t_df['f_pos'])]
                        t_df = t_df[t_index]
                        dic1 = t_df.set_index('f_pos').T.to_dict('list')
                        try:
                            w_pos5= list(t_df[t_df[pvalue_mode]==max(t_df[pvalue_mode])]['f_pos'])[0]
                        except:
                            w_pos5='none'
                            p_t= dic[w][0]
                        else:
                            p_t= dic[w][0]*pos_w1+ dic1[w_pos5][0]*pos_w2
                        finally:
                            if p_t>p_max_t:
                                p_max_t= p_t
                                w1_max=w
                                w2_max= w_pos5
            if p_max_t<=p_min:
                p_max_t=0.
                w1_max,w2_max='none','none'
            return w1_max,w2_max,p_max_t

        t_dic={}
        for t in range(5): t_dic.update({'pos'+str(t+1):tags_t[t]})
        n_list= [int(t!='none') for t in tags_t][1:5]
        p_max=0.
        if (n_list == [1,1,1,0]) | (n_list==[0,0,1,0]) | (n_list==[0,1,1,0]):
            t_df= tags_pos4[tags_pos4['pos']==t_dic['pos4']]
            t_df = t_df[t_df[pvalue_mode] >= p_min]
            if len(t_df) > 0:
                t_index= [ww not in kws_pos5m for ww in list(t_df['f_pos'])]
                t_df= t_df[t_index]
                try:
                    t_dic['pos5']= list(t_df[t_df[pvalue_mode]==max(t_df[pvalue_mode])]['f_pos'])[0]
                except:
                    t_dic['pos5']='none'
                else:
                    p_max=max(t_df[pvalue_mode])
        elif n_list == [1,0,1,1]:
            w_max,p_max= get_tag_101(tags_pos2,tags_pos3,'pos2','pos4',t_dic['pos2'],t_dic['pos4'])
            t_dic['pos3']=w_max
        elif n_list ==[1,1,0,1]:
            w_max, p_max= get_tag_101(tags_pos3,tags_pos4,'pos3','pos5',t_dic['pos3'],t_dic['pos5'])
            t_dic['pos4']=w_max
        elif n_list==[1,0,1,0]:
            w_max, p_max= get_tag_101(tags_pos2,tags_pos3,'pos2','pos4',t_dic['pos2'],t_dic['pos4'])
            if w_max!='none':
                t_dic['pos3']=w_max
                t_df = tags_pos4[tags_pos4['pos'] == t_dic['pos4']]
                t_df = t_df[t_df[pvalue_mode] >= p_min]
                if len(t_df) > 0:
                    t_index = [ww not in kws_pos5m for ww in list(t_df['f_pos'])]
                    t_df = t_df[t_index]
                    try:
                        t_dic['pos5'] =list(t_df[t_df[pvalue_mode] == max(t_df[pvalue_mode])]['f_pos'])[0]
                    except:
                        t_dic['pos5']='none'
                    else:
                        p_max=p_max+max(t_df[pvalue_mode])
        elif n_list==[1,0,0,1]:
            w_pos3,w_pos4,p_max= get_tag_1001(tags_pos2,tags_pos3,tags_pos4,'pos2','pos5')
            t_dic['pos3']= w_pos3
            t_dic['pos4']= w_pos4
        elif n_list == [1,1,0,0]:
            w_pos4,w_pos5,p_max= get_tag_1100(tags_pos3,tags_pos4,'pos3')
            t_dic['pos4']= w_pos4
            t_dic['pos5']= w_pos5

        jobtype_re_pmax.append(p_max)
        jobtype_re2.append(''.join([t_dic['pos' + str(l + 1)] for l in range(5)]).replace('none', ''))
        tags_pos.append([t_dic['pos'+str(l+1)] for l in range(5)])
    tags_pos= np.array(tags_pos)
    jobtype_df = pd.concat(
        [jobtype_df, pd.DataFrame({'jobtype_fix': jobtype_re, 'jobtype_refine': jobtype_re2,'refine_p':jobtype_re_pmax},
                                  index=jobtype_df.index),
         pd.DataFrame(data=tags_pos, columns=['pos1','pos2','pos3','pos4','pos5'])],axis=1)

    jobtype_df.to_csv('jobname_jobtype_by_max_pv_' + tags_mode + "_" + pvalue_mode + '_refine.csv', index=False,
                      encoding='gb18030')



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


#  标准岗位POS关键词根据概率插补，并机器自动标注标准岗位，生成训练标准岗位库
def jobtype_refine():

    test_t= pd.read_csv("jobname_jobtype.csv")

    # refine pos5 which is 'none'
    dd= pd.DataFrame({'pos2':[],'pos5':[],'count':[],'p':[]})
    for p,d in test_t.groupby(['pos2']):
        t=len(d)
        d1= d['pos5'].groupby(d['pos5']).count()
        d2= pd.DataFrame({'pos2':p,'pos5':d1.index,'count':d1})
        d2= pd.DataFrame.reset_index(d2, drop=True)
        #d3= pd.DataFrame({'pos2':p,'pos5':'NA','count':t-sum(d1)}, index=['na'])
        #d2= pd.concat([d2,d3], axis=0, ignore_index=True)
        p1= [round(float(d2['count'][i])/t,4) for i in range(len(d2))]
        p2= pd.DataFrame({'p':p1})
        d2= pd.concat([d2,p2], axis=1)
        dd= pd.concat([dd,d2], axis=0)
    #dd.to_csv("../"+indus[n]+"/data/test.csv",index=False)

    dd = dd[dd['pos5'] == 'none']
    dd = pd.DataFrame.reset_index(dd, drop=True)
    for i in range(len(dd)):
        t = test_t[test_t['pos2'] == dd['pos2'][i]]
        d1 = t[t['pos5'] == 'none']
        d1 = pd.DataFrame.reset_index(d1, drop=True)
        for j in range(len(d1)):
            d2 = t[(t['pos3'] == d1['pos3'][j]) & (t['pos4'] == d1['pos4'][j])]
            d3 = kws_freq_p(d2, 'pos5')
            if (d3['pos5'][0] != 'none') & (d3['p'][0] > 0.4):
                test_t.loc[[x for x in test_t[(test_t['pos2'] == dd['pos2'][i]) & \
                                              (test_t['pos3'] == d1['pos3'][j]) & \
                                              (test_t['pos4'] == d1['pos4'][j]) & \
                                              (test_t['pos5'] == 'none')].index], 'pos5'] = d3['pos5'][0]
            else:
                d2 = t[t['pos4'] == d1['pos4'][j]]
                d3 = kws_freq_p(d2, 'pos5')
                if (d3['pos5'][0] != 'none') & (d3['p'][0] > 0.4):
                    test_t.loc[[x for x in test_t[(test_t['pos2'] == dd['pos2'][i]) & \
                                                  (test_t['pos3'] == d1['pos3'][j]) & \
                                                  (test_t['pos4'] == d1['pos4'][j]) & \
                                                  (test_t['pos5'] == 'none')].index], 'pos5'] = d3['pos5'][0]
    print('pos5 is refined!')
    # refine pos4 which is 'none'
    dd = pd.DataFrame({'pos2': [], 'pos4': [], 'count': [], 'p': []})
    for p, d in test_t.groupby(['pos2']):
        t = len(d)
        d1 = d['pos4'].groupby(d['pos4']).count()
        d2 = pd.DataFrame({'pos2': p, 'pos4': d1.index, 'count': d1})
        d2 = pd.DataFrame.reset_index(d2, drop=True)
        p1 = [round(float(d2['count'][i]) / t, 4) for i in range(len(d2))]
        p2 = pd.DataFrame({'p': p1})
        d2 = pd.concat([d2, p2], axis=1)
        dd = pd.concat([dd, d2], axis=0)
    dd.to_csv("test.csv", index=False)

    dd = dd[dd['pos4'] == 'none']
    dd = pd.DataFrame.reset_index(dd, drop=True)
    for i in range(len(dd)):
        t = test_t[test_t['pos2'] == dd['pos2'][i]]
        d1 = t[t['pos4'] == 'none']
        d1 = pd.DataFrame.reset_index(d1, drop=True)
        for j in range(len(d1)):
            d2 = t[(t['pos3'] == d1['pos3'][j]) & (t['pos5'] == d1['pos5'][j])]
            d3 = kws_freq_p(d2, 'pos4')
            if (d3['pos4'][0] != 'none') & (d3['p'][0] > 0.4):
                test_t.loc[[x for x in test_t[(test_t['pos2'] == dd['pos2'][i]) & \
                                              (test_t['pos3'] == d1['pos3'][j]) & \
                                              (test_t['pos5'] == d1['pos5'][j]) & \
                                              (test_t['pos4'] == 'none')].index], 'pos4'] = d3['pos4'][0]
    print('pos4 is refined!')
    new_jobtype=[]
    for i in range(len(test_t)):
        tmp=''
        for j in range(5):
            tmp=tmp+test_t['pos'+str(j+1)][i]
        tmp= tmp.replace('none','')
        new_jobtype.append(tmp)
    new_jobtype= pd.DataFrame({'jobtype_refine':new_jobtype})
    test_t= pd.concat([test_t,new_jobtype], axis=1)
    test_t.to_csv("jobname_jobtype_refine.csv",index=False)



def kws_freq_p(data,item):
    d1 = data[item].groupby(data[item]).count()
    d2 = pd.DataFrame({item: d1.index, 'count': d1})
    d2 = pd.DataFrame.reset_index(d2, drop=True)
    p1 = [round(float(d2['count'][i]) / sum(d2['count']), 4) for i in range(len(d2))]
    p2 = pd.DataFrame({'p': p1})
    d2 = pd.concat([d2, p2], axis=1)
    d2 = d2.sort_values(by=['p'],ascending= False)
    d2 = pd.DataFrame.reset_index(d2, drop=True)
    return d2


def create_jobtype_database(model_name,jobtypefile,update):
    model= Word2Vec.load(model_name)
    jobtype_t= pd.read_csv(jobtypefile, encoding='GB18030')
    jobtypes, counts,j_vec_sum=[],[],[]
    pos=[]
    vec_ts=np.zeros([1,model.vector_size])
    for x,rs in jobtype_t.groupby('jobtype_refine'):
        if x!='':
            jobtypes.append(x)
            counts.append(len(rs))
            postmp=[]
            vec_t= np.zeros([1,model.vector_size])
            for j in range(5):
                postmp.append(rs['pos'+str(j+1)][rs.index[0]])
                if rs['pos'+str(j+1)][rs.index[0]]!='none':
                    try:
                        vec_t= vec_t+ model.wv.word_vec(rs['pos'+str(j+1)][rs.index[0]])
                    except BaseException as e:
                        print e.message
            vec_ts= np.vstack((vec_ts,vec_t))
            pos.append(postmp)
            j_vec_sum.append(sum(vec_t[0,]))
    vec_ts= vec_ts[1:vec_ts.shape[0]]
    pos_df= pd.DataFrame(data=np.array(pos), columns=['pos1','pos2','pos3','pos4','pos5'])
    if update==0:
        np.save('jobtype_database_vec', vec_ts)
        j_id = ['j' + str(n) for n in np.arange(1000001, 1000001 + len(jobtypes), 1)]
        jobtype_database = pd.DataFrame({'id': j_id, 'jobtype': jobtypes, 'freqs': counts, 'vec_sum': j_vec_sum})
        jobtype_database = pd.concat([jobtype_database,pos_df], axis=1)
        jobtype_database.to_csv('jobtype_database.csv', index=False, encoding='gb18030')
    else:
        jobtype_database= pd.read_csv('jobtype_database.csv', encoding='gb18030')
        id_max= int(max(list(jobtype_database['id']))[1:])
        j_database= list(jobtype_database['jobtype'])
        vec_ts_base= np.load('jobtype_database_vec.npy')
        for i in range(len(jobtypes)):
            if jobtypes[i] not in j_database:
                id_max += 1
                tmp= pd.DataFrame({'id': ['j'+str(id_max)], 'jobtype': [jobtypes[i]], 'freqs':[counts[i]], 'vec_sum': [j_vec_sum[i]]})
                tmp = pd.concat([tmp, pos_df.loc[i:i,:]], axis=1)
                jobtype_database= pd.concat([jobtype_database,tmp],axis=0)
                vec_ts_base= np.concatenate((vec_ts_base,[vec_ts[i]]))
            else:
                jobtype_database.loc[jobtype_database['jobtype']==jobtypes[i],'freqs']= jobtype_database.loc[jobtype_database['jobtype']==jobtypes[i],'freqs']+counts[i]
        np.save('jobtype_database_vec', vec_ts_base)
        jobtype_database.to_csv('jobtype_database.csv', index=False, encoding='gb18030')

    print 'jobtype database OK!'



if __name__=='__main__':

    SQL_con = ["DSN=yxxy-sql;UID=sa;PWD=yxxy",
               "DSN=jobdata;UID=sa;PWD=0829",
               "DSN=SQL_server;UID=sa;PWD=yxxy",
               "DSN=SQL;UID=sa;PWD=yxxy"]

    SQL_par = SQL_con[1]
    stopwords = stopwordslist('jobtype_stop_words.txt')
    jieba.load_userdict('jobtype_dic.txt')

    '''
    print 'Step1: Read test jobname data, create jobtype tags and jobtype name', time.asctime()
    for nn in range(1):
        con_s= sql_jobname(var='jobname', from_table='joblist',keyword='',nums=20000)
        wr= open('test_corpus.txt','a')
        for t in con_s:
            wr.writelines(t[0]+'\n')
        wr.close()
        con_s = [[t.split('\n')[0].decode('utf-8')] for t in open('test_corpus.txt', 'r').readlines()]
        create_jobtype3(jobname_data=con_s,pvalue_mode='p_value_tpos', tags_mode='M4')
        print('       Dataset' + str(nn + 1) + ' completed!')
    '''
    print 'Step2: Auto tagging jobtype name ,create trainning data', time.asctime()
    #jobtype_refine()

    print 'Step1: Read test jobname data, create jobtype tags and jobtype name',time.asctime()
    con_s= [ [t.split('\n')[0].decode('utf-8')] for t in open('test_corpus.txt','r').readlines()]
    #create_jobtype3(jobname_data=con_s, pvalue_mode='p_value_pos', tags_mode='M4')
    #create_jobtype3(jobname_data=con_s, pvalue_mode='p_value_fpos', tags_mode='M4')
    create_jobtype3(jobname_data=con_s, pvalue_mode='p_value_tpos', tags_mode='M4')

    print 'Step2: Auto tagging jobtype name ,create trainning data',time.asctime()
    #jobtype_refine()
    jobtype_refine2(jobtype_df='jobname_jobtype_by_max_pv_M4_p_value_tpos.csv',
                    tags_mode='M4',pvalue_mode='p_value_tpos',pos_w1=0.75, pos_w2=0.25,p_min=0.24)

    print 'Step3: Create jobtype database',time.asctime()
    create_jobtype_database(model_name ='jobtype_model0', jobtypefile='jobname_jobtype_by_max_pv_M4_p_value_tpos_refine.csv', update=0)
    print 'Complete ', time.asctime()
