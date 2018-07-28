#Objetivo: procurar as variáveis com maior correlação e plotar alguns gráficos analíticos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.stats import norm, skew
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) # limitando os floats a tres casas decimais
#Agora carregamos os arquivos

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print('O conjunto de treinamento antes desse preprocessamento possui {}'.format(train.shape) )
print('O conjunto de testes antes desse preprocessamento possui {}'.format(test.shape) )

#Excluímos a coluna ID, pois nao é necessária para o processo de predição. Porém, vamos salvar seus valores antes.
trainID = train['Id']
testID = test['Id']
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)

#Agora vamos procurar e remover os Outliers
# fig,ax = plt.subplots()
# ax.scatter(x=train['GrLivArea'], y=train['SalePrice'])
# plt.ylabel('SalePrice', fontsize=13)
# plt.xlabel('GrLivArea', fontsize=13)
# plt.savefig('objective-research/GrLivArea_SalePrice.png')

#Removemos os outliers cuja grlivarea>4ke saleprice < 300k
train=train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index )
# fig,ax = plt.subplots()
# ax.scatter(x=train['GrLivArea'], y=train['SalePrice'])
# plt.ylabel('SalePrice', fontsize=13)
# plt.xlabel('GrLivArea', fontsize=13)
# plt.savefig('objective-research/GrLivArea_SalePrice_WITHOUT_OUTLIERS.png')

#Agora vamos analisar a variável SalePrice
#Partindo do princípio que já temos o histograma da análise subjetiva, vamos plotar um gráfico ordenado e ver como está distribuída a variável de saída
# sns.distplot(train['SalePrice'], fit=norm)
# fig = plt.figure()
# res = stats.probplot(train['SalePrice'], plot=plt)
# plt.savefig('objective-research/ProbPlot_Initial.png')
# plt.clf()
# plt.cla()

#Vimos que a variavel nao é normalmente distribuida. 
# #Então vamos deixá-la mais linearizada aplicando a função log1p do numpy
train['SalePrice'] = np.log1p(train['SalePrice'])
#Geramos um novo histograma e um novo gráfico de probabilidades
# histo = sns.distplot(train['SalePrice'], fit=norm)
# histofig = histo.get_figure()
# histofig.savefig('objective-research/Histogram-Normally-Distributed.png')
# fig = plt.figure()
# res = stats.probplot(train['SalePrice'], plot=plt)
# plt.savefig('objective-research/ProbPlot_NormallyDistributed.png')


# Agora vamos analisar os dados em branco. 
# Para tal, vamos unir os dois conjuntos de dados em um mesmo dataframe
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values #armazenamos os valores para os dados de treinamento
full_data = pd.concat((train,test)).reset_index(drop=True)
full_data.drop(['SalePrice'], axis=1, inplace=True) #Removemos a coluna SalePrice, pois nao devemos ter contato com o SalePrice do conjunto de testes.
#Feito a união dos conjuntos
full_data_na = (full_data.isnull().sum() /len(full_data)) * 100
full_data_na = full_data_na.drop(full_data_na[full_data_na==0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' : full_data_na})

#Gerar um gráfico com a porcentagem dos dados faltantes para cada variavel.
# f,ax = plt.subplots(figsize=(15,12))
# plt.xticks(rotation=90)
# sns.barplot(x=full_data_na.index, y=full_data_na)
# plt.xlabel('Atributos', fontsize=15)
# plt.ylabel('Porcentagem de valores faltantes', fontsize=15)
# plt.title('Porcentagem de valores faltantes por atributo', fontsize=20)
# plt.savefig('objective-research/missing_values.png')


#Vamos lidar com os dados faltantes agora.
#PoolQC é o atributo que tem maior porcentagem de dados 
# faltantes, e indica a qualidade da piscina.
#NA significa que não há piscinas. Então podemos preencher com um valor que 
# não seja significativo, como None.
#MiscFeature é a segunda variavel com mais valores faltantes.
## NA em MiscFeature significa que não há nenhuma. Podemos preencher com None
#Alley é a terceira variavel com mais valores faltantes
## NA em Alley significa que não há nenhum acesso por becos. Podemos preencher com None.
# O mesmo para variáveis como Fence (cerca) e Fireplace (lareira)
# Também serão preenchidos com None as variáveis categóricas relacionadas a garagem e a porão
for attr in ('PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','MasVnrType', 'MSSubClass'):
    full_data[attr] = full_data[attr].fillna('None')

#GarageYrBlt, GarageArea e GarageCars - se não temos garagem, esses valores devem ser zero. O mesmo para variáveis de Basement (porão)
for attr in ('GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):
    full_data[attr] = full_data[attr].fillna(0)

#Experimento - LotFrontage significa a distancia entre a rua e a casa. 
# Assumindo a similaridade dos lotes da vizinhança, podemos preencher com as médias.
full_data["LotFrontage"] = full_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))


#Plotar um gráfico para provar a análise de Utilities:
# f,ax = plt.subplots(figsize=(15,12))
# plt.xticks(rotation=90)
# sns.countplot(x='Utilities', data=full_data)
# plt.xlabel('Valores de Utilities', fontsize=15)
# plt.ylabel('Porcentagem de valores faltantes', fontsize=15)
# plt.title('Valores para variável Utilities', fontsize=20)
# plt.savefig('objective-research/utilities_values.png')



# Utilities não nos ajuda a predizer, pois aproximadamente 100% dos seus valores são AllPub.Os que não são AllPub são NA. No conjunto de testes até temos um terceiro valor, porém não ajuda para os testes.
# Portanto vamos remover esse atributo
full_data = full_data.drop(['Utilities'], axis=1)





#Vamos montar agora um gráfico intermediário.
full_data_na = (full_data.isnull().sum() /len(full_data)) * 100
full_data_na = full_data_na.drop(full_data_na[full_data_na==0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' : full_data_na})
# f,ax = plt.subplots(figsize=(15,12))
# plt.xticks(rotation=90)
# sns.barplot(x=full_data_na.index, y=full_data_na)
# plt.xlabel('Atributos', fontsize=15)
# plt.ylabel('Porcentagem de valores faltantes', fontsize=15)
# plt.title('Porcentagem de valores faltantes por atributo', fontsize=20)
# plt.savefig('objective-research/missing_values_intermediary.png')

#Vamos tratar MSZoning
f,ax = plt.subplots(figsize=(15,12))
plt.xticks(rotation=90)
sns.countplot(x='MSZoning', data=full_data)
plt.xlabel('Categorias de MSZoning', fontsize=15)
plt.ylabel('Quantidade', fontsize=15)
plt.title('Valores para variável MSZoning', fontsize=20)
plt.savefig('objective-research/mszoning_values.png')

# Assim como Electrical, KitchenQual,Saletype, Exterior1st, Exterior2nd
# Temos uma quantidade ínfima de valores faltantes e a maioria esmagadora dos dados possui a mesma string. Podemos substituir pela string mais comum
for attr in ('MSZoning','KitchenQual', 'SaleType', 'Exterior1st', 'Exterior2nd', 'Electrical'):
    full_data[attr] = full_data[attr].fillna(full_data[attr].mode()[0])

#Resta um atributo
# Functional, quando possui NA, significa Typical segundo a descrição dos dados.
full_data["Functional"] = full_data["Functional"].fillna("Typ")

full_data_na = (full_data.isnull().sum() /len(full_data)) * 100
full_data_na = full_data_na.drop(full_data_na[full_data_na==0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' : full_data_na})
#print(missing_data.head()) # Empty.

# Perfeito. Agora vamos converter valores numéricos em categóricos
full_data['MSSubClass'] = full_data['MSSubClass'].apply(str)
categ = ['OverallCond','YrSold','MoSold']
for i in categ:
    full_data[i] = full_data[i].astype(str)

# Verificamos os tipos das variáveis
# for i in full_data:
#     print(i + " - " + str(type(full_data[i].iloc[0])))

# Agora vamos ver a assinetria das variáveis numéricas. Para tal, vamos filtrá-las
numeric_indexes = full_data.select_dtypes(exclude = ["object"]).columns
categoric_indexes = full_data.select_dtypes(include = ["object"]).columns
#print(numeric_categ)
numeric_columns = full_data[numeric_indexes]
categoric_columns = full_data[categoric_indexes]
skewness = numeric_columns.apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.5] #selecionamos apenas as assimetrias maiores que >0.5
skewed_indexes = skewness.index
numeric_columns[skewed_indexes] = np.log1p(numeric_columns[skewed_indexes])


# proximo passo é verificar as colunas categóricas e separá-las com get_dummies
categoric_columns = pd.get_dummies(categoric_columns)
#Concatenamos
full_data=pd.concat([numeric_columns,categoric_columns], axis=1) #modificamos para ficar com as colunas tratadas
print('O conjunto de total após o preprocessamento possui {}'.format(full_data.shape) )

# Agora dividimos os datasets novamente.
train = full_data[:ntrain]
test = full_data[ntrain:]
print('O conjunto de treinamento antes desse preprocessamento possui {}'.format(train.shape) )
print('O conjunto de testes antes desse preprocessamento possui {}'.format(test.shape) )

# Vamos aplicar alguns modelos.
#Lasso.

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

# Aproveitamos as instanciacoes
elastic = make_pipeline(RobustScaler(), ElasticNet(alpha=0.005, l1_ratio=.9, random_state=3))
ker_ridge = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
gboost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =5)
xgboost = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05, max_depth=3, min_child_weight=1.7817, n_estimators=2200,reg_alpha=0.4640, reg_lambda=0.8571,subsample=0.5213, silent=1,random_state =7, nthread = -1)
lgbm = model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,learning_rate=0.05, n_estimators=720,max_bin = 55, bagging_fraction = 0.8,bagging_freq = 5, feature_fraction = 0.2319,feature_fraction_seed=9, bagging_seed=9,min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

from utils import AveragingModels

#Improve1
#avg_model = AveragingModels((lasso, lgbm, gboost,xgboost))

#Improve2
avg_model = AveragingModels((lasso, lgbm, gboost,xgboost, ker_ridge, elastic))
avg_model.fit(train.values, y_train)
avg_model_predictions = np.expm1(avg_model.predict(test.values))
avg_model_output = 'Id,SalePrice\n'
for i in range(0,len(avg_model_predictions)):
    avg_model_output = avg_model_output + str(testID[i])+ ','+str(avg_model_predictions[i])+'\n'
avg_model_file = open('avg_model_submission_2.csv','w')
avg_model_file.write(avg_model_output)
avg_model_file.close()