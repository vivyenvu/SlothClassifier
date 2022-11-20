import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics, model_selection

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

df_chars = df_chars[['claw_length_cm', 'size_cm', 'tail_length_cm', 'weight_kg', 'sub_specie']]
print(list(df_chars.columns.values))
mylog_model = linear_model.LogisticRegression(max_iter=250)
y = df_chars.values[:200, 4]
x = df_chars.values[:200, 0:4]


mylog_model.fit(x, y)

print(mylog_model.predict([[6.514, 64.194, 5.8, 6.635]]))

y_pred = mylog_model.predict(x)
print(metrics.accuracy_score(y, y_pred))


#mylog_model3 = linear_model.LogisticRegression(max_iter=5000)
#y3 = df_chars.values[:, 4]
#x3 = df_chars.values[:, 0:4]
#mylog_model3.fit(x3, y3)
#print(mylog_model3.predict([[6.514, 64.194, 5.8, 6.635]]))


#df_chars = df_chars[['claw_length_cm', 'weight_kg', 'sub_specie']]
#mylog_model2 = linear_model.LogisticRegression(max_iter=5000)
#y2 = df_chars.values[:, 2]
#x2 = df_chars.values[:, :2]
#mylog_model2.fit(x2, y2)
#print(mylog_model2.predict([[6.514, 6.635]]))



