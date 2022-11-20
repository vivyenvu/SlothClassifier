import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics, model_selection

# Load csv data
df = pd.read_csv('sloth_data.csv')

# Delete unused columns
del df["Unnamed: 0"]
df_chars = df.drop(columns=['endangered', 'specie'])

# Create scatter matrix of data
scatter_matrix(df)
plt.show()

# Create histogram of data
df.hist()
plt.show()

# Create boxplot of sizes for each of the 3 endangered classifications
endangered_filter = df['endangered'] == 'critically_endangered'
dfE = df.loc[endangered_filter, :]
least_concern = df.loc[df['endangered'] == 'least_concern', 'size_cm'].values
vulnerable = df.loc[df['endangered'] == 'vulnerable', 'size_cm'].values
critically_endangered = df.loc[df['endangered'] == 'critically_endangered', 'size_cm'].values

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
ax.boxplot([least_concern, vulnerable, critically_endangered],
           labels=['Least concern', 'Vulnerable', 'Critically Endangered'])
ax.set_ylabel('Size in cm')
plt.show()

# Reorder columns
df_chars = df_chars[['claw_length_cm', 'size_cm', 'tail_length_cm', 'weight_kg', 'sub_specie']]

# Train model to identify sloth subspecies based on anthropometrics
model = linear_model.LogisticRegression(max_iter=5001, solver='newton-cg')
y = df_chars.values[:, 4]
x = df_chars.values[:, 0:4]
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.5)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Calculate the model's accuracy
score = metrics.accuracy_score(y_test, y_pred)
print('Model accuracy is: ' + str(score))

# UI
print(
    'This program will help you identify the subspecies of your sloth. Please enter values up to 3 decimal places (eg. 8.147) ')
while True:
    claw_input = input('What is the claw length in cm? ')
    try:
        claw_input = float(claw_input)
    except ValueError:
        print('Error: Invalid input. Restart program and try again. ')
        exit()

    size_input = input("What is the sloth's size in cm? ")
    try:
        size_input = float(size_input)
    except ValueError:
        print('Error: Invalid input. Restart program and try again. ')
        exit()

    tail_input = input('What is the tail length in cm? ')
    try:
        tail_input = float(tail_input)
    except ValueError:
        print('Error: Invalid input. Restart program and try again. ')
        exit()

    weight_input = input('What is the weight in kg? ')
    try:
        weight_input = float(weight_input)
    except ValueError:
        print('Error: Invalid input. Restart program and try again. ')
        exit()

    print('Thank you for all that information. Your sloth may be a: ')
    print(model.predict(
        [[np.float64(claw_input), np.float64(size_input), np.float64(tail_input), np.float64(weight_input)]]))
    exit()
