"""
Bu uygulama icin gerekli veri seti:
    https://www.kaggle.com/shaunthesheep/microsoft-catsvsdogs-dataset
adresinden indirilmelidir. Veri seti archive adli klasor ile bu kod dosyasiyla 
ayni dizine konumlandirilabilir.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm
import os
from sklearn.metrics import classification_report,confusion_matrix
from PIL import Image
from tensorflow.keras.applications import ResNet152V2
from sklearn.model_selection import train_test_split
from warnings import filterwarnings
filterwarnings('ignore')

labels = ['Cat','Dog']
image_size = 64
X_train = []
y_train = []
for i in labels:
    folderPath = os.path.join('archive/PetImages/'+i)
    klasor=folderPath+"/"
    for j in tqdm(os.listdir(folderPath)):
        yol=os.path.join(klasor,j)
        try:
            img=np.array(Image.open(yol).convert('RGB').resize((image_size, image_size), Image.ANTIALIAS))
            X_train.append(img)
            y_train.append(i)
        except:
            continue
        
X_train=np.array(X_train)
y_train=np.array(y_train)       
X_train,X_test,y_train,y_test = train_test_split(X_train,y_train, test_size=0.1,random_state=101)
def kodlama (y_t):
    y_new = []
    for i in y_t:
        y_new.append(labels.index(i))
    return tf.keras.utils.to_categorical(y_new)
y_train=kodlama(y_train)
y_test=kodlama(y_test)

resnet = ResNet152V2(weights='imagenet', include_top=False,input_shape=(image_size,image_size,3))
model = resnet.output
model = tf.keras.layers.GlobalAveragePooling2D()(model)
model = tf.keras.layers.Dropout(rate=0.5)(model)
model = tf.keras.layers.Dense(2,activation='softmax')(model)
model = tf.keras.models.Model(inputs=resnet.input, outputs = model)
model.summary()

model.compile(loss='categorical_crossentropy',optimizer = 'Adam', metrics= ['accuracy'])
model.fit(X_train,y_train,validation_split=0.1, epochs =5, verbose=1, batch_size=64)

pred = model.predict(X_test)
pred = np.argmax(pred,axis=1)
y_test_new = np.argmax(y_test,axis=1)

print(classification_report(y_test_new,pred))
fig,ax=plt.subplots(1,1,figsize=(14,7))
sns.heatmap(confusion_matrix(y_test_new,pred),ax=ax,xticklabels=labels,yticklabels=labels,annot=True)
plt.show()