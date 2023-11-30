import numpy as np
import pandas as pd
v =pd.read_csv("hayvanatbahcesi.csv",encoding='unicode_escape')
veri = v.copy()

from sklearn import preprocessing
sayisallastirma = preprocessing.LabelEncoder()

veri["hayvan adi"]=sayisallastirma.fit_transform(veri["hayvan adi"])

girisler=np.array(veri.drop(["sinifi"],axis=1))
cikis=np.array(veri["sinifi"])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(girisler,cikis, test_size=0.2,random_state=109)

from sklearn.naive_bayes import CategoricalNB

gnb = CategoricalNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
cm= confusion_matrix(y_test,y_pred)
index = ['1','2','3','4','5'] 
columns =  ['1','2','3','4','5'] 
cm_df = pd.DataFrame(cm,columns,index)                      
plt.figure(figsize=(10,6))  
sns.heatmap(cm_df, annot=True,fmt="d")

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
