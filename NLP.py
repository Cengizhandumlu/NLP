import numpy as np
import pandas as pd

yorumlar=pd.read_csv('Restaurant_Reviews.csv')

import re

""" stopword = anlamsız kelimeler the gibi, bunları temizleyecegimiz biryapı var """

import nltk

""" turkish stop word icin kaynak var trstop"""
""" bu stopwordleri indirip pythonda liste olarak yüklememiz yeterli"""

nltk.download('stopwords')

from nltk.corpus import stopwords


""" kelimeleri köklerine ayıracagız """

from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()

#Preprocessing(Önişleme)
derlem=[]

for i in range(1000):
    """ noktaların kaldırılması """
    yorum=re.sub('[^a-zA-Z]',' ',yorumlar['Review'][i])
    """ büyük kücük harf problemi """
    """ yazının tamamını kücük harfe ceviriyor"""    
    yorum=yorum.lower()
    """ yazılan kelimeleri python listesine cevirmek istiyorum."""
    yorum=yorum.split()
    yorum=[ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
    yorum = ' '.join(yorum)
    derlem.append(yorum)
    
"""CountVectorizer sayac vektörü """

#Feauture Extraction (öznitelik çıkarması)
#Bag of Words (BOW)

from sklearn.feature_extraction.text import CountVectorizer


cv=CountVectorizer(max_features=2000) #en fazla kullanılan 2000 kelimeyi al

X=cv.fit_transform(derlem).toarray() #bagımsız degisken
y=yorumlar.iloc[:,1].values #bagımlı degisken

#Makine Öğrenmesi

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

from sklearn.naive_bayes import GaussianNB

gnb=GaussianNB()

gnb.fit(X_train,y_train)

y_pred=gnb.predict(X_test)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)

print(cm)































