# imported libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

# load both training and test sets
train = pd.read_csv('housing_train.csv')
test = pd.read_csv('housing_test.csv')
n = train.shape[0]
y_train = train.SalePrice.values

# get dataframe of all missing data percentages
all_data = pd.concat((train, test)).reset_index(drop = True)
all_data.drop(['SalePrice'], axis = 1, inplace = True)
all_data_missing = (all_data.isnull().sum() / len(all_data)) * 100
all_data_missing = all_data_missing.drop(all_data_missing[all_data_missing == 0].index).sort_values(ascending = False)[:30]
missing = pd.DataFrame({'Percent Missing' : all_data_missing})

print(missing)

# fill in missing data or drop
all_data['PoolQC'] = all_data['PoolQC'].fillna('None')

all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')

all_data['Alley'] = all_data['Alley'].fillna('None')

all_data['Fence'] = all_data['Fence'].fillna('None')

all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('None')

all_data['YrBlt'] = all_data['Alley'].fillna(0)

all_data['LotFrontage'] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x : x.fillna(x.median()))

for i in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[i] = all_data[i].fillna('None')

for i in ('GarageArea', 'GarageCars', 'GarageYrBlt'):
    all_data[i] = all_data[i].fillna(0)
    
for i in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[i] = all_data[i].fillna(0)

for i in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[i] = all_data[i].fillna('None')
    
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

all_data["MSZoning"] = all_data["MSZoning"].fillna(all_data['MSZoning'].mode()[0])

all_data["Functional"] = all_data["Functional"].fillna("Typ")

all_data = all_data.drop(['Utilities'], axis = 1)

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
    
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

# check missing values again to make sure all were filled in or dropped
all_data_missing = (all_data.isnull().sum() / len(all_data)) * 100
all_data_missing = all_data_missing.drop(all_data_missing[all_data_missing == 0].index).sort_values(ascending = False)[:30]
missing = pd.DataFrame({'Percent Missing' : all_data_missing})

print()
print(missing)

# Changing numerical and categorical variables
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)

all_data['OverallCond'] = all_data['OverallCond'].astype(str)

all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

# Adding a feature to represent total square footage

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF'] 

all_data = pd.get_dummies(all_data)

# transform the data by applying log to make the data fit the normal curve better

train['SalePrice'] = np.log(train['SalePrice'])

train = all_data[:train.shape[0]]
test = all_data[train.shape[0]:]

X_train, X_test, y_train, y_test = train_test_split(train, y_train, test_size=0.1, random_state = 0)

forest = RandomForestRegressor(n_estimators= 300, random_state=0)
forest.fit(X_train, y_train)
predictions = forest.predict(X_test)
print(forest.score(X_train, y_train))

gradientBoosting = GradientBoostingRegressor(n_estimators= 300, random_state=0)
forest.fit(X_train, y_train)
predictions = forest.predict(X_test)
print(forest.score(X_train, y_train))

plt.figure(figsize=(12,8))
plt.plot(y_test, color='red')
plt.plot(predictions, color='blue')
plt.show()

test_predictions = forest.predict(test)

submission = pd.DataFrame({'Id': test.Id, 'SalePrice': test_predictions})
submission.to_csv('submission.csv', index = False)
