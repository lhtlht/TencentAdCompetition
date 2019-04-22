import pandas as pd
import numpy as np

from feature_processing import *
from model import *


def load_data(is_offline):
    train_dtypes = {'origid':'uint32', 'bid':'int16'}
    train = pd.read_csv(totalExposureLog_path2, encoding="utf-8", dtype=train_dtypes) #02/16-03/19
    train = train[train['requestDate'] >= '2019-02-20']
    test = pd.read_csv(test_sample_path2, encoding="utf-8")
    #数据拼接处理
    train = data_join(train)
    train.fillna(0, inplace=True)
    train[['createTimeYear','createTimeMonth', 'createTimeDay', 'createTimeHour']] = train[['createTimeYear','createTimeMonth', 'createTimeDay', 'createTimeHour']].astype(np.int8)
    train[['origid', 'bid', 'accountid']] = train[
        ['origid', 'bid', 'accountid']].astype(np.int32)
    train[['shoptype', 'size']] = train[
        ['shoptype', 'size']].astype(np.int8)

    model_train = train
    model_test = train[train['requestDate']=='2019-03-19']
    if is_offline:
        train_label = model_train['showNum']
        test_label = model_test['showNum']
        model_train = model_train.drop(columns=['showNum'])
        model_test = model_test.drop(columns=['showNum'])
        return model_train, model_test, train_label, test_label
    else:
        train = train[train['requestDate'] >= '2019-03-10']
        train_label = train['showNum']
        test_label = 1
        train = train.drop(columns=['showNum'])
        return train, test, train_label, test_label

def data_join(train):
    ad_operation = pd.read_csv(ad_operation_path2, encoding="utf-8")
    ad_static_feature = pd.read_csv(ad_static_feature_path2, encoding="utf-8")
    train = train.merge(ad_static_feature, how="left", on=["origid"]) #putTime usergroup 50个为空
    train.dropna(subset=['createTimeDate'],inplace=True)
    train = train.merge(ad_operation, how="left", on=["origid","bid"])

    return train

def eval_model(y, y_hat, bid):
    eval_df = pd.DataFrame()
    eval_df['y'] = y
    eval_df['y_hat'] = y_hat
    eval_df['bid'] = bid
    eval_df['accuracy'] = eval_df.apply(lambda x: abs(x['y']-x['y_hat'])/(x['y']+x['y_hat'])*2, axis=1)
    accuracy = eval_df['accuracy'].mean()
    return accuracy

if __name__ == "__main__":
    is_offline = False
    train, test, train_label, test_label = load_data(is_offline)
    train = model_feature_processing(train)
    print(train.info())
    print(train.columns)
    print(train.shape)
    print(test.columns)

    #特征处理

    #模型训练
    model_type = 'lgb'
    label_features = []
    onehot_features = ['accountid', 'shoptype']
    features = ['createTimeDay', 'createTimeWeek', 'createTimeYear', 'createTimeMonth', 'createTimeHour',
                'requestDateWeek',
                'bid', 'size'] + onehot_features
    onehot_features = []
    print(train.shape)
    preds = reg_model(train, test, train_label, model_type, onehot_features, label_features, features)



    if is_offline:
        accuracy = eval_model(test_label, preds, 1)
        print("accuracy:", accuracy)
        #0.53
    else:
        df = pd.DataFrame()
        df['id'] = test['id']
        pp = []
        for i in preds:
            if i<0:
                pp.append(0)
            else:
                pp.append(round(i,4))
        df['y'] = pp
        df.to_csv("./data/submissionA/0422.csv", index=False, header=None)


