import matplotlib.pyplot as plt
import pandas as pd

'''
df = pd.read_csv('dataset_outer.csv')

df1 = df1.to_dict()
plt.bar(*zip(df1.items()))
plt.show()

print(len(df.index) - df.apply(lambda x : x.isnull().sum(), axis='rows'))
x = len(df.index) - df.apply(lambda x : x.isnull().sum(), axis='rows')

x.to_csv('size.csv')
'''

df = pd.read_csv('sizes.csv')


df1 = df[df['Dias'] < 1000].dropna()
# print(df1)
print(len(df1.index))

df2 = df[(df['Dias'] >= 1000) & (df['Dias'] < 2000)].dropna()
# print(df2)
print(len(df2.index))

df3 = df[(df['Dias'] >= 2000) & (df['Dias'] < 3000)].dropna()
# print(df3)
print(len(df3.index))

df4 = df[(df['Dias'] >= 3000) & (df['Dias'] < 4000)].dropna()
# print(df4)
print(len(df4.index))



df5 = df[(df['Dias'] >= 4000) & (df['Dias'] < 5000)].dropna()
# print(df5)
print(len(df5.index))

df6 = df[(df['Dias'] >= 5000)].dropna()
# print(df6)
print(len(df6.index))

print(df.nsmallest(10,'Dias'))