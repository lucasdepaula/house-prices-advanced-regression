#Objetivo: entender o funcionamento do dataset, o significado de suas colunas e fazer análises superficiais
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


data_train = pd.read_csv('data/train.csv')
def corr_matrix(data):
    corrmat = data.corr()
    f,ax = plt.subplots(figsize=(12,9))
    sns.heatmap(corrmat,vmax=1,square=True)
    plt.savefig('human-research/corr.png')
    plt.cla()
    plt.clf()

def histogram(col):
    histo = sns.distplot(data_train[col])
    fig = histo.get_figure()
    fig.savefig('human-research/Histogram_'+col+'.png')
    #fig.cla()
    fig.clf()

def scatter(col):
    data = pd.concat([data_train['SalePrice'], data_train[col]],axis=1)
    data.plot.scatter(x=col,y='SalePrice', ylim=(0,800000)) # limite maximo baseado no describe
    plt.savefig('human-research/Scatter_SalePrice_'+col+'.png')
    plt.cla()
    plt.clf()

def boxplot(col):
    data = pd.concat([data_train['SalePrice'], data_train[col]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=col, y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800000)
    plt.savefig('human-research/BoxPlot_'+col+'.png')
    plt.cla()
    plt.clf()
#print(data_train.columns)

#Analisando a variavel de saída SalePrice

print(data_train['SalePrice'].describe())
# Matriz Correlação
corr_matrix(data_train)

#Histograma das variáveis
# for i in data_train.columns:
#     histogram(i)
histogram('SalePrice')

# Variáveis com Potencial de Influencia (baseado em análise humana)
numeric_cols = ['LotArea', 'TotalBsmtSF', 'GrLivArea']
categoric_cols = ['Neighborhood','Condition1', 'Condition2','OverallQual','OverallCond']

for i in numeric_cols:
    scatter(i)

# Análise das variáveis categóricas
for i in categoric_cols:
    boxplot(i)