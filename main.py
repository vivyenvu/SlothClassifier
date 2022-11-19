import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('sloth_data.csv')
del df["Unnamed: 0"]
#print(df['endangered'].value_counts())

df_chars = df.drop(columns=['endangered', 'specie'])
#print('This is df')
#print(df)
#print('This is df_chars')
#print(df_chars)

#scatter_matrix(df)
#plt.show()

#df.hist()
#plt.show()

endangered_filter = df['endangered'] == 'critically_endangered'
dfE = df.loc[endangered_filter, :]
#print(dfE['endangered'].value_counts())

#least_concern = df.loc[df['endangered']=='least_concern', 'size_cm'].values
#vulnerable = df.loc[df['endangered']=='vulnerable', 'size_cm'].values
#critically_endangered = df.loc[df['endangered']=='critically_endangered', 'size_cm'].values
#plt.boxplot([least_concern, vulnerable, critically_endangered], labels=['Least concern','Vulnerable', 'Critically Endangered'])
#plt.show()

#df_chars = df_chars[['claw_length_cm', 'size_cm', 'tail_length_cm', 'weight_kg', 'sub_specie']]
#print(list(df_chars.columns.values))
#mylog_model = linear_model.LogisticRegression(max_iter=250)
#y = df_chars.values[:200, 4]
#x = df_chars.values[:200,0:4]

#mylog_model.fit(x, y)

#print(mylog_model.predict([[9.2, 64.7, -.01, 6.6]]))

df_chars = df_chars[['claw_length_cm', 'weight_kg', 'sub_specie']]
print(list(df_chars.columns.values))
mylog_model = linear_model.LogisticRegression(tol = 0.1)
y = df_chars.values[:, 2]
x = df_chars.values[:,0:2]

mylog_model.fit(x, y)

print(mylog_model.predict([[8.596, 3.487]]))



