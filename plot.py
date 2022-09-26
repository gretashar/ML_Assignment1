import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for statistical data visualization
# %matplotlib inline

import warnings
from SVM import svm_test


warnings.filterwarnings('ignore')

data = "./data/heart.csv"

df = pd.read_csv(data)
print(df.shape)
print(df.head())
col_names = df.columns
print(col_names)
print(df['output'].value_counts())
print(df['output'].value_counts()/np.float(len(df)))
print(df.info())
print(df.isnull().sum())
print(round(df.describe(),2))


plt.figure(figsize=(24,20))

plt.subplot(4, 2, 1)
fig = df.boxplot(column='age')
fig.set_title('')
fig.set_ylabel('Age')

plt.subplot(4, 2, 2)
fig = df.boxplot(column='sex')
fig.set_title('')
fig.set_ylabel('Sex')

plt.subplot(4, 2, 3)
fig = df.boxplot(column='cp')
fig.set_title('')
fig.set_ylabel('CP')

plt.subplot(4, 2, 4)
fig = df.boxplot(column='trtbps')
fig.set_title('')
fig.set_ylabel('Trtbps')

plt.subplot(4, 2, 5)
fig = df.boxplot(column='chol')
fig.set_title('')
fig.set_ylabel('Chol')

plt.subplot(4, 2, 6)
fig = df.boxplot(column='fbs')
fig.set_title('')
fig.set_ylabel('FBS')

plt.subplot(4, 2, 7)
fig = df.boxplot(column='restecg')
fig.set_title('')
fig.set_ylabel('Restecg')

plt.subplot(4, 2, 8)
fig = df.boxplot(column='thalachh')
fig.set_title('')
fig.set_ylabel('Thalachh')

plt.savefig('./plots/Heartbox1.png')
plt.clf()


plt.subplot(4, 2, 1)
fig = df.boxplot(column='exng')
fig.set_title('')
fig.set_ylabel('Exng')

plt.subplot(4, 2, 2)
fig = df.boxplot(column='oldpeak')
fig.set_title('')
fig.set_ylabel('Oldpeak')


plt.subplot(4, 2, 3)
fig = df.boxplot(column='slp')
fig.set_title('')
fig.set_ylabel('SLP')


plt.subplot(4, 2, 4)
fig = df.boxplot(column='caa')
fig.set_title('')
fig.set_ylabel('CAA')


plt.subplot(4, 2, 5)
fig = df.boxplot(column='thall')
fig.set_title('')
fig.set_ylabel('Thall')

plt.subplot(4, 2, 6)
fig = df.boxplot(column='output')
fig.set_title('')
fig.set_ylabel('Output')

plt.savefig('./plots/Heartbox2.png')
plt.clf() 



plt.figure(figsize=(24,20))

plt.subplot(4, 2, 1)
fig = df["age"].hist(bins=20)
fig.set_title('')
fig.set_ylabel('Age')

plt.subplot(4, 2, 2)
fig = df["sex"].hist(bins=2)
fig.set_title('')
fig.set_ylabel('Sex')

plt.subplot(4, 2, 3)
fig = df["cp"].hist(bins=4)
fig.set_title('')
fig.set_ylabel('CP')

plt.subplot(4, 2, 4)
fig = df["trtbps"].hist(bins=20)
fig.set_title('')
fig.set_ylabel('Trtbps')

plt.subplot(4, 2, 5)
fig = df["chol"].hist(bins=20)
fig.set_title('')
fig.set_ylabel('Chol')

plt.subplot(4, 2, 6)
fig = df["fbs"].hist(bins=2)
fig.set_title('')
fig.set_ylabel('FBS')

plt.subplot(4, 2, 7)
fig = df["restecg"].hist(bins=3)
fig.set_title('')
fig.set_ylabel('Restecg')

plt.subplot(4, 2, 8)
fig = df["thalachh"].hist(bins=20)
fig.set_title('')
fig.set_ylabel('Thalachh')

plt.savefig('./plots/Hearthist1.png')
plt.clf()


plt.subplot(4, 2, 1)
fig = df["exng"].hist(bins=20)
fig.set_title('')
fig.set_ylabel('Exng')

plt.subplot(4, 2, 2)
fig = df["oldpeak"].hist(bins=20)
fig.set_title('')
fig.set_ylabel('Oldpeak')


plt.subplot(4, 2, 3)
fig = df["slp"].hist(bins=20)
fig.set_title('')
fig.set_ylabel('SLP')


plt.subplot(4, 2, 4)
fig = df["caa"].hist(bins=20)
fig.set_title('')
fig.set_ylabel('CAA')


plt.subplot(4, 2, 5)
fig = df["thall"].hist(bins=20)
fig.set_title('')
fig.set_ylabel('Thall')

plt.subplot(4, 2, 6)
fig = df["output"].hist(bins=20)
fig.set_title('')
fig.set_ylabel('Output')

plt.savefig('./plots/Hearthist2.png')

svm_test(df)



