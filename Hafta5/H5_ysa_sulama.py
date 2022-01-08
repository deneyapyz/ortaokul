from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics 

dataset = np.genfromtxt('INTELLIGENT IRRIGATION SYSTEM.csv', delimiter=',')
Giris=dataset[1:,0:2]
Cikis =dataset[1:,2]
Giris_train, Giris_test, Cikis_train, Cikis_test = train_test_split(Giris, Cikis, test_size=0.2, random_state=0)

model = Sequential()
model.add(Dense(6, input_dim=2, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(Giris_train, Cikis_train, epochs=30, batch_size=5)
Cikis_pred = model.predict(Giris_test)
Cikis_pred=(Cikis_pred>0.5).flatten()
print("Doğruluk:",metrics.accuracy_score(Cikis_test, Cikis_pred))



from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
cm= confusion_matrix(Cikis_test,Cikis_pred)
index = ['Çalışmıyor','Çalışıyor'] 
columns =  ['Çalışmıyor','Çalışıyor'] 
cm_df = pd.DataFrame(cm,columns,index)                      
plt.figure(figsize=(10,6))  
sns.heatmap(cm_df, annot=True,fmt="d")

