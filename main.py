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
mylog_model = linear_model.LogisticRegression(max_iter=5001, solver='newton-cg')
y = df_chars.values[:, 4]
x = df_chars.values[:, 0:4]
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3)
mylog_model.fit(x_train, y_train)
print(mylog_model.predict([[7.262, 62.939, 3.405, 7.042]]))
y_pred = mylog_model.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred))



