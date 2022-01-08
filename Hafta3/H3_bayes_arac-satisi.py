import numpy as np
import pandas as pd
veri =pd.read_csv("car.csv",encoding='unicode_escape')
veri_copy=veri.copy()

from sklearn import preprocessing
sayisallastirma = preprocessing.LabelEncoder()

veri_copy["fiyat"]=sayisallastirma.fit_transform(veri_copy["fiyat"])
veri_copy["onarim"]=sayisallastirma.fit_transform(veri_copy["onarim"])
veri_copy["kapi sayisi"]=sayisallastirma.fit_transform(veri_copy["kapi sayisi"])
veri_copy["kisi sayisi"]=sayisallastirma.fit_transform(veri_copy["kisi sayisi"])
veri_copy["bagaj boyutu"]=sayisallastirma.fit_transform(veri_copy["bagaj boyutu"])
veri_copy["Guvenlik"]=sayisallastirma.fit_transform(veri_copy["Guvenlik"])
veri_copy["satis "]=sayisallastirma.fit_transform(veri_copy["satis "])


girisler=np.array(veri_copy.drop(["satis "],axis=1))
cikis=np.array(veri_copy["satis "])

from sklearn.model_selection import train_test_split
giris_egitim, giris_test, satis_egitim, satis_test = train_test_split(girisler,cikis, test_size=0.3,random_state=109)


from sklearn.naive_bayes import CategoricalNB
model = CategoricalNB()
model.fit(giris_egitim, satis_egitim)
satis_tahmin= model.predict(giris_test)


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
cm= confusion_matrix(satis_test,satis_tahmin)
index = ['Kolay','Normal','Zor','Çok kolay'] 
columns = ['Kolay','Normal','Zor','Çok kolay'] 
cm_df = pd.DataFrame(cm,columns,index)                      
plt.figure(figsize=(10,6))  
sns.heatmap(cm_df, annot=True,fmt="d")


from sklearn import metrics
print("Modelin Doğruluğu:",metrics.accuracy_score(satis_test, satis_tahmin))




















