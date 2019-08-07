from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from Bio.Data import IUPACData

def Seq2comp(seq):
  ind=[]
  for aa in IUPACData.protein_letters:
    ac=seq.count(aa)
    ind.append(ac)
  return np.array(ind)/20
col_pred='Immunogenicity'

train=pd.read_csv('data/train.csv')

#df=pd.read_csv('data/train.csv')
#pos=len(df[df[col_pred] == 1])
#neg=len(df[df[col_pred] == 0])
#df1 = df.loc[df[col_pred] == 0].sample(neg-pos)
#train = df.loc[~df.index.isin(df1.index)]
col=(list(IUPACData.protein_letters))
train['feature']=train['Peptide'].apply(Seq2comp)
train[col]=pd.DataFrame(train.feature.values.tolist(), index= train.index)

test=pd.read_csv('data/test.csv')
test['feature']=test['Peptide'].apply(Seq2comp)
test[col]=pd.DataFrame(test.feature.values.tolist(), index= test.index)


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(20,)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train[col].to_numpy(),train[col_pred].to_numpy(),epochs=2)

pmids=test['PubMed ID'].unique()
pmid_perf={}
for pmid in pmids:
  test_pmid=test.loc[test['PubMed ID']==pmid]
  test_loss, test_acc = model.evaluate(test_pmid[col].to_numpy(), test_pmid[col_pred].to_numpy())
  pmid_perf[pmid]=test_acc
  print("Test performance for ",pmid, " = ", test_acc)
print("Average performance ", np.array(list(pmid_perf.values())).mean())

