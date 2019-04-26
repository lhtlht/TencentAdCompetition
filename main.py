import pandas as pd
import numpy as np

from feature_processing import *
from model import *


def load_data():
    train = pd.read_csv(totalExposureLog_path4, encoding="utf-8") #02/16-03/19
    test = pd.read_csv(test_sample_path2, encoding="utf-8")
    ad_operation = pd.read_csv(ad_operation_path2, encoding="utf-8")
    ad_static_feature = pd.read_csv(ad_static_feature_path2, encoding="utf-8")
    return train, test, ad_operation, ad_static_feature



def eval_model(y, y_hat, bid):
    eval_df = pd.DataFrame()
    eval_df['y'] = y
    eval_df['y_hat'] = y_hat
    eval_df['bid'] = bid
    eval_df['accuracy'] = eval_df.apply(lambda x: 2*abs(x['y']-x['y_hat'])/(abs(x['y'])+abs(x['y_hat'])), axis=1)
    accuracy = eval_df['accuracy'].mean()
    return accuracy

if __name__ == "__main__":

    train, test, ad_operation, ad_static_feature = load_data()
    is_offline = True
    #数据的拼接
    train = train.merge(ad_static_feature, how="left", on=["origid"])
    train.dropna(subset=['createTimeDate'], inplace=True)
    #train.fillna(-1, inplace=True)
    train.fillna('-1', inplace=True)
    #ad_operation = ad_operation.merge(ad_static_feature[['origid','createTimeDate']], how='left', on=['origid'])
    #ad_operation['ModifyTimeDate'] = ad_operation.apply(lambda x: x['createTimeDate'] if x['operationType']==2 else x['ModifyTimeDate'], axis=1)
    #ad_operation.fillna(0, inplace=True)
    #train = train.merge(ad_operation, how="left", left_on=["origid", "bid", "requestDate"], right_on=["origid", "bid", "ModifyTimeDate"])
    print(train.info())

    if is_offline:
        test = train[train['requestDate'] == '2019-03-19']
        train = train[train['requestDate'] != '2019-03-19']

    print(train.info())
    print(train.columns)
    print(train.shape)
    print(test.columns)
    #特征处理
    """
     'maxbid', 'maxshow', 'minbid', 'minshow', 'meanshow', 'showPNum'
     
    """
    train, test = model_feature_processing(train, test)
    train_label = train['showNum']

    #模型训练

    model_type = 'lgb'
    label_features = ['accountid', 'shoptype', 'origid', 'industryid',
                      'createTimeDay', 'createTimeWeek', 'createTimeYear', 'createTimeMonth', 'createTimeHour']
    onehot_features = ['accountid', 'shoptype', 'origid','shopid', 'industryid',
                       'createTimeDay', 'createTimeWeek', 'createTimeYear', 'createTimeMonth', 'createTimeHour','requestDateWeek']
    features = ['bid', 'size',
                'maxbid', 'maxshow', 'minbid', 'minshow', 'meanshow', 'showPNum'
                ]
    #onehot_features = []
    print(train.shape)
    print(train.info())
    preds = reg_model(train, test, train_label, model_type, onehot_features, label_features, features)



    if is_offline:
        test_label = test['showNum']
        accuracy = eval_model(test_label, preds, 1)
        print("accuracy:", accuracy)
        #accuracy: 0.22218013123411592
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
        df.to_csv("./data/submissionA/0425.csv", index=False, header=None)


