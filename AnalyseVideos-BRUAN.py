from multiprocessing import Process, Manager
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import measure
import datetime
import pandas as pd
import numpy as np
import time
import re
from IPython.display import Image
from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_file("Translation API-b9a2439f440d.json")
import glob, os
import pytesseract
from google.cloud import translate
from skimage.metrics import structural_similarity as compare_ssim
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
try:
    from PIL import Image
except ImportError:
    import Image
import sys
from nltk import tokenize
from operator import itemgetter
import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
stop_words = set(stopwords.words('english'))

import spacy
# from spacy import display
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

#Count vectorizer for N grams
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


brands = ["Braun","Conair","Panasonic","Philips","Remington","Wahl","Gillette","Idea Village","SEB","Datsumo","Ke-non","Silk'n","Ya-man","Rei Beaute","Norelco"]

# Return Frame run time
def get_frame_runtime(frameNumber,fps):
    return int(frameNumber/fps)

def get_imp_words(doc=None):
    total_words = doc.split()
    total_word_length = len(total_words)

    total_sentences = tokenize.sent_tokenize(doc)
    total_sent_len = len(total_sentences)

    tf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.','')
        if each_word not in stop_words:
            if each_word in tf_score:
                tf_score[each_word] += 1
            else:
                tf_score[each_word] = 1

    tf_score.update((x, y/int(total_word_length)) for x, y in tf_score.items())

    def check_sent(word, sentences): 
        final = [all([w in x for w in word]) for x in sentences] 
        sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
        return int(len(sent_len))

    idf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.','')
        if each_word not in stop_words:
            if each_word in idf_score:
                idf_score[each_word] = check_sent(each_word, total_sentences)
            else:
                idf_score[each_word] = 1

    idf_score.update((x, math.log(int(total_sent_len)/y)) for x, y in idf_score.items())

    tf_idf_score = {key: tf_score[key] * idf_score.get(key, 0) for key in tf_score.keys()}

    def get_top_n(dict_elem, n):
        result = dict(sorted(dict_elem.items(), key = itemgetter(1), reverse = True)[:n]) 
        return result

    return(get_top_n(tf_idf_score, 1000))

def get_unique_frames(raw_frames,fps):
    frame1 = raw_frames[0]
    frame2 = raw_frames[1]
    uniqueframes = []
    frameNo = []
    text = []
    time = []
    prevframe = 0
    i=0
    while i < len(raw_frames):
        (score, diff) = compare_ssim(frame1, raw_frames[i], full=True)
        if score > 0.80:
            pass
        else:
            frame1 = Image_PreProcess(frame1)
            uniqueframes.append(frame1)
            text.append(analyse_frame(frame1))
            frameNo.append(prevframe)
            time.append(get_frame_runtime(prevframe,fps))
            prevframe = i
            frame1 = raw_frames[i]
        i = i+1
    return uniqueframes,frameNo,text,time

def analyse_frame(frame):
    output_text = pytesseract.image_to_string(Image.fromarray(frame), lang='eng')
    output_text = output_text.replace('\n','')
    return output_text

def Image_PreProcess(image):
    th3 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    return th3

def get_Brands(df=None):
    df["Brands"] = ""
    for idx in df.index:
        try:
            all_brands = []
            for w1 in word_tokenize(df["text"][idx]):
                for w2 in brands:
                    if levenshtein_ratio_and_distance(w1,w2,True) > 0.75:
                        all_brands.append(w2)
            df["Brands"][idx] = all_brands
        except:
            pass
    return df

def ngrams_top(corpus,ngram_range,n=None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.
    """
    vec = CountVectorizer(stop_words = 'english',ngram_range=ngram_range).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    total_list=words_freq[:n]
    df=pd.DataFrame(total_list,columns=['text','count'])
    return df

def levenshtein_ratio_and_distance(s, t, ratio_calc = False):
    s = s.lower()
    t = t.lower()
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)


    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k
 
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0
            else:
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row-1][col] + 1,      
                                 distance[row][col-1] + 1,          
                                 distance[row-1][col-1] + cost)     
    if ratio_calc == True:
        Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
        return Ratio
    else:
        return "The strings are {} edits away".format(distance[row][col])

def analyse_brands_video(fileName):
    if fileName.split("\\")[-1].split('_')[-1].split('.')[0] == 'screen':
        PaticipantName = fileName.split("\\")[-1].split('_')[0]
        cap = cv2.VideoCapture(fileName)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        raw_frames = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            raw_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        uf = get_unique_frames(raw_frames,fps)
        df = pd.DataFrame(list(zip(uf[0],uf[1],uf[2],uf[3])),columns=("Image","FrameNumber","text","Time(Sec)"))
        df = get_Brands(df)
        if not(df.empty):
            df = df.join(pd.get_dummies(df["Brands"].apply(pd.Series).stack()).sum(level=0))
            df = df.drop(["Image","FrameNumber","text","Brands"],axis=1)
            df["ParticipantName"] = PaticipantName
            df.to_csv("OutPut/"+PaticipantName+".csv",index=False)

def multiprocessing_func(x):
    time.sleep(2)
    print(x)
#     print('{} is {} number'.format(x, is_prime(x)))


def divide_chunks(l, n):
# looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]
        
        
        
if __name__ == '__main__':
    starttime = time.time()
    processes = []
    path = 'C:\\Users\\aateam\\Desktop\\Loop11 Video Download\\Downloaded\\Prod-Braun-English-Online Search Activity Male\\'
#     path = "Video files/"
    mylist = [f for f in glob.glob(path+'*.mp4')]
    
    n = 2
    x = list(divide_chunks(mylist, n))
    
    for x1 in tqdm(x):
        for i in x1:
            p = Process(target=analyse_brands_video, args=(i,))
            processes.append(p)
            p.start()

        for process in processes:
            process.join()
        
    print()    
    print('Time taken = {} seconds'.format(time.time() - starttime))
    
#     Desktop/Loop11 Video Download/Downloaded/