#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics
import os

# hide warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


housing_data = pd.read_csv("train.csv", encoding = 'utf-8')
housing_data.head(10)
housing_data.shape


# In[3]:


housing_data.describe()


# In[4]:


housing_data.isnull().sum()


# In[5]:


sns.heatmap(housing_data.isnull(),yticklabels= False,cbar=False)


# In[6]:


### Finding the percentage of nulls in the columns
housing_data.columns[housing_data.isnull().any()] 

housing_data_null = housing_data.isnull().sum()/len(housing_data)*100
housing_data_null = housing_data_null[housing_data_null>0]
housing_data_null.sort_values(inplace=True, ascending=False)


# In[7]:


housing_data_null


# In[8]:


### Put None in categorical columns
cols_to_update = ["Alley", "MasVnrType", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]
for i in cols_to_update:
    housing_data[i].fillna("none", inplace=True)


# In[9]:


### Any more nulls to worry about
housing_data.columns[housing_data.isnull().any()] 

housing_data_null_chck = housing_data.isnull().sum()/len(housing_data)*100
housing_data_null_chck = housing_data_null_chck[housing_data_null_chck>0]
housing_data_null_chck.sort_values(inplace=True, ascending=False)


# In[10]:


housing_data_null_chck


# In[11]:


### "LotFrontage" is Linear feet of street connected to property. it can be imputed with similar 'Neighborhood' values
housing_data['LotFrontage'] = housing_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))


# In[12]:


### Can fill with median values for below 2 columns with no impact
housing_data["GarageYrBlt"].fillna(housing_data["GarageYrBlt"].median(), inplace=True)
housing_data["MasVnrArea"].fillna(housing_data["MasVnrArea"].median(), inplace=True)


# In[13]:


### dropping Electrical column as it has no significance
housing_data["Electrical"].dropna(inplace=True)


# In[14]:


### Dropping ID column and analyzing first the numeric columns
housing_data_num = housing_data.select_dtypes(include=['float64', 'int64'])
housing_data_num.head()
housing_data_num.drop(['Id'],axis=1,inplace=True)


# In[15]:


housing_data.shape


# In[16]:


#  'Sale Price' wrt 'Neighborhood' to understand target variable changes with few independant variables

plt.figure(figsize=(20, 8))
sns.barplot(x="Neighborhood", y="SalePrice", data= housing_data)
plt.title("Sales Price wrt Neighbourhood")
plt.xticks(rotation=90)


# In[17]:


# Same for 'overall condition' wrt 'Saleprice'

plt.figure(figsize=(20, 8))
sns.barplot(x="OverallCond", y="SalePrice", data= housing_data)
plt.title("Sales Price wrt Overall Condition")
plt.xticks(rotation=90)


# In[18]:


plt.figure(figsize=(20, 8))
sns.barplot(x="OverallQual", y="SalePrice", data= housing_data)
plt.title("Sales Price wrt Overall Quality")
plt.xticks(rotation=90)


# In[19]:


### Shown above is that Increase in the overall quality has a direct positive effect on the sale price


# In[20]:


sns.distplot(housing_data['SalePrice'])


# In[21]:


### Above distribution plot is skewed towards left so attempting now log transformation


# In[22]:


housing_data['SalePrice']=np.log1p(housing_data['SalePrice'])


# In[23]:


sns.distplot(housing_data['SalePrice'])


# In[24]:


### As now centralized distribution is seen as above so let's check correlation matrix
cor = housing_data_num.corr()
cor


# In[25]:


### Checking with heat map as well
plt.figure(figsize=(30,20))

# heatmap
sns.heatmap(cor, cmap="YlGnBu", annot=True)
plt.show()


# In[26]:


# Checking now with a pairplot 
sns.set()
cols = ['SalePrice', 'GrLivArea', 'GarageCars', 'BsmtUnfSF', 'BsmtFinSF1', 'GarageArea', 'TotalBsmtSF', 'YearBuilt', 'TotRmsAbvGrd', 'GarageYrBlt']
sns.pairplot(housing_data[cols], size = 2.5)
plt.show()


# In[27]:


### Variables are correlated so dropping them after analyzing them via pair plot as above
housing_data = housing_data.drop(['GarageCars'], axis = 1)
housing_data = housing_data.drop(['BsmtUnfSF'], axis = 1)
housing_data = housing_data.drop(['TotRmsAbvGrd'], axis = 1)
housing_data = housing_data.drop(['GarageYrBlt'], axis = 1) 


# In[28]:


housing_data.select_dtypes(exclude=['object']) ### Numeric checks


# In[29]:


# prAge  --> Property Age
housing_data['prAge'] = (housing_data['YrSold'] - housing_data['YearBuilt'])
housing_data.head()


# In[30]:


### Checking  numeric columns
sns.jointplot(x='GrLivArea', y='SalePrice', data=housing_data)
plt.show()


# In[31]:


### Removing outliers


# In[32]:


q1 = housing_data['GrLivArea'].quantile(0.25)
q3 = housing_data['GrLivArea'].quantile(0.75)
value = q3-q1
lower_value  = q1-1.5*value
higher_value = q3+1.5*value
housing_data= housing_data[(housing_data['GrLivArea']<higher_value) & (housing_data['GrLivArea']>lower_value)]


# In[33]:


housing_data.shape


# In[34]:


# 1stFlrSF vs SalePrice
sns.jointplot(x = housing_data['1stFlrSF'], y = housing_data['SalePrice'])
plt.show()


# In[35]:


sns.jointplot(x = housing_data['2ndFlrSF'], y = housing_data['SalePrice'])
plt.show()


# In[36]:


### First level houses i.e. '0' and second floor per Sq.Ft have  a steady increase


# In[37]:


### Dropping columns having very less variance and significance eg. Id
housing_data = housing_data.drop(['Id'], axis=1)
housing_data = housing_data.drop(['Street'], axis = 1)
housing_data = housing_data.drop(['Utilities'], axis = 1)


# In[38]:


### As Age of property is now derived so dropping the below columns


# In[39]:


housing_data = housing_data.drop(['MoSold'], axis = 1)
housing_data = housing_data.drop(['YrSold'], axis = 1)
housing_data = housing_data.drop(['YearBuilt'], axis = 1)
housing_data = housing_data.drop(['YearRemodAdd'], axis = 1)


# In[ ]:





# In[40]:


housing_data = housing_data.drop(['PoolQC','MiscVal', 'Alley', 'RoofMatl', 'Condition2', 'Heating', 'GarageCond', 'Fence', 'Functional' ], axis = 1)


# In[ ]:





# In[41]:


#type of each feature in data: int, float, object
types = housing_data.dtypes
#numerical values are either type int or float
numeric_type = types[(types == 'int64') | (types == float)] 
#categorical values are type object
categorical_type = types[types == object]


# In[42]:


pd.DataFrame(types).reset_index().set_index(0).reset_index()[0].value_counts()


# In[43]:


numerical_columns = list(numeric_type.index)


# In[44]:


categorical_columns = list(categorical_type.index)


# In[45]:


print(numerical_columns)


# In[46]:


print(categorical_columns)


# In[47]:


housing_data = pd.get_dummies(housing_data, drop_first=True )
housing_data.head()


# In[48]:


X = housing_data.drop(['SalePrice'], axis=1)

X.head()


# In[49]:


# Putting response variable to y
y = housing_data['SalePrice']

y.head()


# In[50]:


# Splitting the data into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=50)


# In[ ]:





# In[51]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'Fireplaces', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'prAge']] = scaler.fit_transform(X_train[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'Fireplaces', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'prAge']])

X_test[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'Fireplaces', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'prAge']] = scaler.fit_transform(X_test[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'Fireplaces', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'prAge']])


# In[ ]:





# In[ ]:





# In[ ]:





# In[52]:


# Applying Lasso

# list of alphas to tune
params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}
lasso = Lasso()

# cross validation
folds = 5
model_cv = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            

model_cv.fit(X_train, y_train) 


# In[53]:


# cv_results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results = cv_results[cv_results['param_alpha']<=1]
cv_results.head()


# In[54]:


# plotting mean test and train scoes with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')

# plotting
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')
plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper left')
plt.show()


# In[55]:


# At alpha = 0.01, even the smallest of negative coefficients that have some predictive power towards 'SalePrice' have been generated

alpha = 0.01
lasso = Lasso(alpha=alpha)

lasso.fit(X_train, y_train)
lasso.coef_


# In[56]:


# lasso model parameters
model_parameters = list(lasso.coef_ )
model_parameters.insert(0, lasso.intercept_)
model_parameters = [round(x, 3) for x in model_parameters]
cols = X.columns
cols = cols.insert(0, "constant")
list(zip(cols, model_parameters))


# In[57]:


# lasso regression
lm = Lasso(alpha=0.01)
lm.fit(X_train, y_train)

# prediction on the test set(Using R2)
y_train_pred = lm.predict(X_train)
print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))
y_test_pred = lm.predict(X_test)
print(metrics.r2_score(y_true=y_test, y_pred=y_test_pred))


# In[58]:


print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))


# In[59]:


mod = list(zip(cols, model_parameters))
para = pd.DataFrame(mod)
para.columns = ['Variable', 'Coeff']
para.head()


# In[60]:


para = para.sort_values((['Coeff']), axis = 0, ascending = False)
pred = pd.DataFrame(para[(para['Coeff'] != 0)])


# In[61]:


pred


# In[62]:


pred.shape


# In[63]:


print(list(pred['Variable']))


# In[64]:


X_train_lasso = X_train[['GrLivArea', 'OverallQual', 'OverallCond', 'TotalBsmtSF', 'GarageArea', 'BsmtFinSF1', 'Fireplaces', 'LotArea', 'LotFrontage', 'BsmtFullBath', 'Foundation_PConc', 'OpenPorchSF', 'FullBath', 'ScreenPorch', 'WoodDeckSF']]
X_test_lasso = X_test[['GrLivArea', 'OverallQual', 'OverallCond', 'TotalBsmtSF', 'GarageArea', 'BsmtFinSF1', 'Fireplaces', 'LotArea', 'LotFrontage', 'BsmtFullBath', 'Foundation_PConc', 'OpenPorchSF', 'FullBath', 'ScreenPorch', 'WoodDeckSF']]


# In[65]:


X_train_lasso


# In[66]:


X_test_lasso


# In[67]:


### Ridge Regression


# In[68]:


# list of alphas to tune for Ridge Regression
params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}


# In[69]:


ridge = Ridge()

# cross validation
folds = 5
model_cv = GridSearchCV(estimator = ridge, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            
model_cv.fit(X_train, y_train) 


# In[70]:


cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results = cv_results[cv_results['param_alpha']<=5]
cv_results.head()


# In[71]:


# plotting mean test and train scoes with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')

# plotting
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')
plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper left')
plt.show()


# In[72]:


alpha = 2
ridge = Ridge(alpha=alpha)

ridge.fit(X_train, y_train)
ridge.coef_


# In[73]:


# ridge model parameters
model_parameters = list(ridge.coef_)
model_parameters.insert(0, ridge.intercept_)
model_parameters = [round(x, 3) for x in model_parameters]
cols = X.columns
cols = cols.insert(0, "constant")
list(zip(cols, model_parameters))


# In[74]:


# ridge regression
lm = Ridge(alpha=2)
lm.fit(X_train, y_train)

# predict
y_train_pred = lm.predict(X_train)
print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))
y_test_pred = lm.predict(X_test)
print(metrics.r2_score(y_true=y_test, y_pred=y_test_pred))


# In[75]:


print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))


# In[76]:


mod_ridge = list(zip(cols, model_parameters))
paraRFE = pd.DataFrame(mod_ridge)
paraRFE.columns = ['Variable', 'Coeff']
res=paraRFE.sort_values(by=['Coeff'], ascending = False)
res.head(20)
paraRFE = paraRFE.sort_values((['Coeff']), axis = 0, ascending = False)
predRFE = pd.DataFrame(paraRFE[(paraRFE['Coeff'] != 0)])
predRFE


# In[77]:


### Observation:
#### Though the model performance by Ridge Regression was better in terms of R2 values of Train and Test, 
#### it is better to use Lasso, since it brings and assigns a zero value to insignificant features, enabling us to choose
#### the predictive variables.


# In[78]:


pred.set_index(pd.Index(['C','x1', 'x2', 'x3', 'x4', 'x5' , 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16']), inplace = True) 
pred


# ### Equation:
#     
# ####    Log(Y) = C + 0.125(x1) + 0.112(x2) +  0.050(x3) + 0.042(x4) + 0.035(x5) + 0.034(x6) + 0.024(x7) +  0.015(x8) + 0.014(x9) + 0.010(x10)
# ####                + 0.010(x11) + 0.005(x12) - 0.007(x13) - 0.007(x14) - 0.008(x15) - 0.095(x16) + Error term(RSS + alpha * (sum of absolute value of coefficients)
# 
# 
# ### INFERENCE
# #### Suggestion is to keep a check on these predictors affecting the price of the house.
# 
# #### The higher values of positive coeeficients suggest a high sale value.
# 
# #### Some of those features are:-
#  ###  Feature  -  Description  
#  
#  ###  GrLivArea  -  Above grade (ground) living area square feet  
#   ### OverallQual  -  Rates the overall material and finish of the house  
#   ### OverallCond  -  Rates the overall condition of the house  
#   ### TotalBsmtSF   -  Total square feet of basement area  
#   ###  GarageArea    - Size of garage in square feet  
#         
# #### The higher values of negative coeeficients suggest a decrease in sale value.
# 
# #### Some of those features are:-
#      ### Feature  -  Description  
#   
#      ### prAge  -  Age of the property 
#      ### MSSubClass  -  Identifies the type of dwelling involved in the sale
#     
# 
# #### When the market value of the property is lower than the Predicted Sale Price, its the time to buy.
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




