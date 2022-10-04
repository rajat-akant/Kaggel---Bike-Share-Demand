import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import VotingRegressor
import numpy as np
from sklearn.pipeline import Pipeline

######################################### Casual ######################################################################
#Training the Model on Train Set on Casual:
    
#Loading the Training Data Set
train = pd.read_csv(r"C:\PC_Data\DBDA\Machine Learning\Kaggle\Capita Bike Share Demand\train.csv", parse_dates=['datetime'])

#Understanding the Data Types And Null Values
train.info()
#Data is majorly numerical with no null/nan values. Needs pre-processing for the datetime column only.

#Splitting the datetime column
train['year']=train['datetime'].dt.year
train['month']=train['datetime'].dt.month
train['day']=train['datetime'].dt.day
train['hour']=train['datetime'].dt.hour

#Few numerical columns need conversion to categorical for making the model understand data better
cat_cols = ['season', 'holiday', 'workingday', 'weather', 'year', 'month', 'hour']
for col in cat_cols:
    train[col] = train[col].astype('category')

#Encoding the Categorical Columns
dum_train = pd.get_dummies(train)

#Taking "Casual" as target variable
x_train = dum_train.drop(['count','datetime','casual','registered'],axis = 1)
y_train = dum_train['casual']

kfold = KFold(n_splits=5,shuffle=True,random_state=2022)
rf = RandomForestRegressor()
lgbm = LGBMRegressor()

models = [('rf',rf),('lgbm',lgbm)]
voting = VotingRegressor(estimators = models)

params = {'rf__max_depth':[None,3,7,10],
          'rf__min_samples_split':[2,5,8],
          'rf__min_samples_leaf':[1,4,10],
          'lgbm__boosting_type': ['gbdt'],
          'lgbm__learning_rate': [0.05, 0.1],
          'lgbm__n_estimators': [500]}

gcv = GridSearchCV(voting, param_grid=params, scoring = 'neg_root_mean_squared_error',cv=kfold,verbose=3)
gcv.fit(x_train,y_train)
print(gcv.best_params_)#
# =============================================================================
# {'lgbm__boosting_type': 'gbdt', 
#  'lgbm__learning_rate': 0.1, 
#  'lgbm__n_estimators': 500, 
#  'rf__max_depth': None, 
#  'rf__min_samples_leaf': 1, 
#  'rf__min_samples_split': 5}
# =============================================================================
print(gcv.best_score_)# -16.474877858582328
best_esti = gcv.best_estimator_

# Predicting on Test Set with Model Trained on "Casual":

test = pd.read_csv(r"C:\PC_Data\DBDA\Machine Learning\Kaggle\Capita Bike Share Demand\test.csv", parse_dates=['datetime'])

test['year']=test['datetime'].dt.year
test['month']=test['datetime'].dt.month
test['day']=test['datetime'].dt.day
test['hour']=test['datetime'].dt.hour


cat_cols = ['season', 'holiday', 'workingday', 'weather', 'year', 'month', 'hour']
for col in cat_cols:
    test[col] = test[col].astype('category')

dum_test = pd.get_dummies(test)

x_test = dum_test.drop('datetime',axis=1)
results_c = best_esti.predict(x_test)


##################################### Register ###############################################################

#Training the Model with Train Set on Register:
# Taking "Registered" as Target Variable:
x_train = dum_train.drop(['count','datetime','casual','registered'],axis = 1)
y_train = dum_train['registered']

kfold = KFold(n_splits=5,shuffle=True,random_state=2022)
rf = RandomForestRegressor()
lgbm = LGBMRegressor()

models = [('rf',rf),('lgbm',lgbm)]
voting = VotingRegressor(estimators = models)

params = {'rf__max_depth':[None,3,7,10],
          'rf__min_samples_split':[2,5,8],
          'rf__min_samples_leaf':[1,4,10],
          'lgbm__boosting_type': ['gbdt'],
          'lgbm__learning_rate': [0.05, 0.1],
          'lgbm__n_estimators': [500]}

gcv = GridSearchCV(voting, param_grid=params, scoring = 'neg_root_mean_squared_error',cv=kfold,verbose=3)
gcv.fit(x_train,y_train)
print(gcv.best_params_)#
# =============================================================================
# {'lgbm__boosting_type': 'gbdt', 
#  'lgbm__learning_rate': 0.1, 
#  'lgbm__n_estimators': 500, 
#  'rf__max_depth': None, 
#  'rf__min_samples_leaf': 1, 
#  'rf__min_samples_split': 2}
# =============================================================================
print(gcv.best_score_)# -35.478285534491995
best_esti = gcv.best_estimator_

# Predicting on Test Set with model trained on "Registered":

x_test_r = dum_test.drop('datetime',axis=1)
results_r = best_esti.predict(x_test_r)


#Creating Sample Submission:

ss = pd.read_csv(r"C:\PC_Data\DBDA\Machine Learning\Kaggle\Capita Bike Share Demand\sampleSubmission.csv")

results_c[results_c<0]=0
results_r[results_r<0]=0

results_c = np.round(results_c)
results_r = np.round(results_r)

ss['count']= results_c + results_r

ss.to_csv("voting_rf_lgbm.csv",index=False)
