import pandas as pd
import numpy as np
import sys
import math
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from scipy import sparse
from scipy.sparse import csr_matrix
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'n_estimators': 10000,
    'metric': 'mae',
    'learning_rate': 0.01,
    'min_child_samples': 5,
    'min_child_weight': 0.01,
    'subsample_freq': 1,
    'num_leaves': 63,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0,
    'reg_lambda': 5,
    'verbose': -1,
    'random_state': 4590,
    'n_jobs': 4
}
xgb_params = {
        'booster': 'gbtree',
        'learning_rate': 0.01,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.8,
        'objective': 'reg:linear',
        'n_estimators': 10000,
        'min_child_weight': 3,
        'gamma': 0,
        'silent': True,
        'n_jobs': -1,
        'random_state': 4590,
        'reg_alpha': 2,
        'reg_lambda': 0.1,
        'alpha': 1,
        'verbose': 1
    }

def multi_column_LabelEncoder(df,columns,rename=True):
    le = LabelEncoder()
    for column in columns:
        print(column,"LabelEncoder......")
        le.fit(df[column])
        df[column+"_index"] = le.transform(df[column])
        if rename:
            df.drop([column], axis=1, inplace=True)
            df.rename(columns={column+"_index":column}, inplace=True)
    return df

def reg_model(model_train, test, train_label, model_type, onehot_features, label_features, features):
    import lightgbm as lgb
    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestRegressor
    model_train.reset_index(inplace=True)
    train_label.index = range(len(train_label))
    test.reset_index(inplace=True)
    if model_type == 'rf':
        model_train.fillna(0, inplace=True)

    combine = pd.concat([model_train, test], axis=0)
    combine = multi_column_LabelEncoder(combine, label_features, rename=True)
    #one hot 处理
    if onehot_features != []:
        onehoter = OneHotEncoder()
        X_onehot = onehoter.fit_transform(combine[onehot_features])
        train_x_onehot = X_onehot.tocsr()[:model_train.shape[0]].tocsr()
        test_x_onehot = X_onehot.tocsr()[model_train.shape[0]:].tocsr()

        train_x_original = combine[features][:model_train.shape[0]]
        test_x_original = combine[features][model_train.shape[0]:]

        train_x = sparse.hstack((train_x_onehot, train_x_original)).tocsr()
        test_x = sparse.hstack((test_x_onehot, test_x_original)).tocsr()
    else:
        train_x = combine[features][:model_train.shape[0]].values
        test_x = combine[features][model_train.shape[0]:].values


    train_y = train_label

    n_fold = 5
    count_fold = 0
    preds_list = list()
    oof = np.zeros(train_x.shape[0])
    kfolder = KFold(n_splits=n_fold, shuffle=True, random_state=2019)
    kfold = kfolder.split(train_x, train_y)
    for train_index, vali_index in kfold:
        print("training......fold",count_fold)
        count_fold = count_fold + 1
        k_x_train = train_x[train_index]
        k_y_train = train_y.loc[train_index]
        k_x_vali = train_x[vali_index]
        k_y_vali = train_y.loc[vali_index]
        if model_type == 'lgb':
            dtrain = lgb.Dataset(k_x_train, k_y_train)
            dvalid = lgb.Dataset(k_x_vali, k_y_vali, reference=dtrain)
            lgb_model = lgb.LGBMRegressor(**lgb_params)
            lgb_model = lgb_model.fit(k_x_train, k_y_train, eval_set=[(k_x_train, k_y_train), (k_x_vali, k_y_vali)],
                                      early_stopping_rounds=200, verbose=False, eval_metric="l2")
            k_pred = lgb_model.predict(k_x_vali, num_iteration=lgb_model.best_iteration_)
            pred = lgb_model.predict(test_x, num_iteration=lgb_model.best_iteration_)
        elif model_type == 'xgb':
            xgb_model = XGBRegressor(**xgb_params)
            xgb_model = xgb_model.fit(k_x_train, k_y_train, eval_set=[(k_x_train, k_y_train), (k_x_vali, k_y_vali)],
                                      early_stopping_rounds=200, verbose=False)
            k_pred = xgb_model.predict(k_x_vali)
            pred = xgb_model.predict(test_x)
        elif model_type == 'rf':
            rf_model = RandomForestRegressor(n_estimators=100, max_depth=3, criterion="mae",n_jobs=-1,random_state=2019)
            model = rf_model.fit(k_x_train, k_y_train)
            k_pred = rf_model.predict(k_x_vali)
            pred = rf_model.predict(test_x)
        preds_list.append(pred)
        oof[vali_index] = k_pred
    preds_columns = ['preds_{id}'.format(id=i) for i in range(n_fold)]
    preds_df = pd.DataFrame(data=preds_list)
    preds_df = preds_df.T
    preds_df.columns = preds_columns
    preds = list(preds_df.mean(axis=1))


    return preds