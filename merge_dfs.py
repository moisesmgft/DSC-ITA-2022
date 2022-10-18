from os import listdir
import pandas as pd

src_path = 'data/yahoo/sp500/'
files = [src_path + x for x in listdir(src_path) if 'csv' in x]

df1 = pd.read_csv(files[0])
df1 = df1.rename({'Date': 'Dia','Close': files[0].replace(src_path,'').replace('.csv','')}, axis=1)

#print(df1)

for i in range(1,len(files)):
    df2 = pd.read_csv(files[i])
    df2 = df2.rename({'Date': 'Dia','Close': files[i].replace(src_path,'').replace('.csv','')}, axis=1)
    #df1.merge(df2, how='left', left_on='Date', right_on='Date')
    df1 = pd.merge(df1, df2, how='outer', on=['Dia','Dia']) # talvez inner, pq faz a interseccao


df1['Dia'] = pd.to_datetime(df1['Dia'])
df1 = df1.sort_values(by='Dia')
df1.to_csv('data_set.csv')