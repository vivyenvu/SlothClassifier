import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#from matplotlib import pyplot
from sklearn import linear_model

df = pd.read_csv('sloth_data.csv')
del df["Unnamed: 0"]
print(df['endangered'].value_counts())

df_chars = df.drop(columns=['endangered', 'specie', 'sub_specie'])
print(df_chars)

#sns.heatmap(df_chars, annot=True, cmap='Blues');
#plt.ylabel('Actual label')
#plt.xlabel('Predicted label')
#plt.show()

scatter_matrix(df)
plt.show()

#print(df)
df.hist()
plt.show()

endangered_filter = df['endangered'] == 'critically_endangered'
dfE = df.loc[endangered_filter, :]
print(dfE['endangered'].value_counts())

least_concern = df.loc[df['endangered']=='least_concern', 'size_cm'].values
vulnerable = df.loc[df['endangered']=='vulnerable', 'size_cm'].values
critically_endangered = df.loc[df['endangered']=='critically_endangered', 'size_cm'].values
plt.boxplot([least_concern, vulnerable, critically_endangered], labels=['Least concern','Vulnerable', 'Critically Endangered'])
plt.show()


mylog_model = linear_model.LogisticRegression()
y = df_chars.values[:, 3]
x = df_chars.values[:,0:4]

mylog_model.fit(x, y)

#print(mylog_model.predict([[6,9]]))



