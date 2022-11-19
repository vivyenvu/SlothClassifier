import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#from matplotlib import pyplot
from sklearn import linear_model

df = pd.read_csv('sloth_data.csv')
del df["Unnamed: 0"]
print(df['endangered'].value_counts())
#print(df)
df.hist()
plt.show()

endangered_filter = df['endangered'] == 'critically_endangered'
dfE = df.loc[endangered_filter, :]
print(dfE['endangered'].value_counts())
# create deep copy of dataframe
#df_endangered = df.copy(deep=True)
#df_endangered.groupby(['endangered']).value_counts().plot(kind='pie')
#df2 = df_endangered.plot.pie(figsize=(10,10), autopct='$.2f%%', subplots=True)

#print(df_endangered)
#df_endangered.plot(kind='pie', subplots=True)



#mylog_model = linear_model.LogisticRegression()
#y = df.to_numpy()
#x = df.values[:,0:2]

#mylog_model.fit(x, y)

#print(mylog_model.predict([[6,9]]))



