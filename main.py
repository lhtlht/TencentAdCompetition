import pandas as pd
import numpy as np

from feature_processing import *
from model import *


def load_data():
    train = pd.read_csv(totalExposureLog_path4, encoding="utf-8") #02/16-03/19
    test = pd.read_csv(test_sample_path2, encoding="utf-8")
    ad_operation = pd.read_csv(ad_operation_path2, encoding="utf-8")
    #['ModifyTimeDate', 'bid', 'createModifyTime', 'modifyTag','operationType', 'origid', 'putTime', 'usergroup', 'createTimeDate']

    ad_static_feature = pd.read_csv(ad_static_feature_path2, encoding="utf-8")

    origid_size = pd.read_csv(exposure_origid_size_path, encoding="utf-8")
    return train, test, ad_operation, ad_static_feature, origid_size



def eval_model(y, y_hat, bid):
    eval_df = pd.DataFrame()
    eval_df['y'] = y
    eval_df['y_hat'] = y_hat
    eval_df['bid'] = bid
    eval_df['smape'] = eval_df.apply(lambda x: 2*abs(x['y']-x['y_hat'])/(abs(x['y'])+abs(x['y_hat'])), axis=1)
    eval_df['mse'] = eval_df.apply(lambda x: (x['y'] - x['y_hat'])*(x['y'] - x['y_hat']), axis=1)
    smape = eval_df['smape'].mean()
    mse = eval_df['mse'].mean()
    return smape,mse

if __name__ == "__main__":
    train, test, ad_operation, ad_static_feature, origid_size = load_data()
    train = train[train['bid']>0]
    is_offline = False
    #数据的拼接
    train = train.merge(ad_static_feature, how="left", on=["origid"])
    train.drop(['size'], axis=1, inplace=True)
    train = train.merge(origid_size, how="left", on=["origid"])

    train.dropna(subset=['createTimeDate'], inplace=True)
    ad_operation = ad_operation.merge(ad_static_feature[['origid','createTimeDate']], how='left', on=['origid'])
    ad_operation['ModifyTimeDate'] = ad_operation.apply(lambda x: x['createTimeDate'] if x['operationType']==2 else x['ModifyTimeDate'], axis=1)
    ad_operation.drop(['createTimeDate'], axis=1, inplace=True)
    #ad_operation.fillna(0, inplace=True)
    print(train.shape)
    train = train.merge(ad_operation, how="left", left_on=["origid", "bid", "requestDate"], right_on=["origid", "bid", "ModifyTimeDate"])
    #特征处理
    """
     'maxbid', 'maxshow', 'minbid', 'minshow', 'meanshow', 'showPNum'
     
    """
    train.drop(['ModifyTimeDate'], axis=1, inplace=True)
    train.drop(['modifyTag'], axis=1, inplace=True)
    train.drop(['operationType'], axis=1, inplace=True)

    train.fillna(0, inplace=True)

    train, test = model_feature_processing(train, test)
    train = train[train['requestDate'] >= '2019-03-01']
    if is_offline:
        test = train[train['requestDate'] == '2019-03-19']
        train = train[train['requestDate'] != '2019-03-19']
    #训练方式1
    #train_label = train['showNum']
    #训练方式2
    train['per_show'] = train['showNum'] / train['bid']
    train_label = train['per_show']
    test2 = test.copy()
    test.drop(['bid','id', 'showPNum'], axis=1, inplace=True)
    print(test.columns)
    test.to_csv('aaa.csv', index=False)
    test = test.drop_duplicates(subset=['origid'], keep='first')
    print("test2",test2.shape)
    print("test",test.shape)
    #模型训练

    model_type = 'lgb'
    label_features = ['accountid', 'shoptype', 'origid', 'industryid','shopid',
                      'createTimeDay', 'createTimeWeek', 'createTimeYear', 'createTimeMonth', 'createTimeHour']
    onehot_features = ['accountid', 'shoptype', 'origid', 'industryid', 'shopid',
                      'requestDateWeek',
                       'createTimeDay', 'createTimeWeek', 'createTimeYear', 'createTimeMonth', 'createTimeHour',
                       'diffdays','diffmonths']
    features = [ 'size', 'maxbid', 'maxshow', 'minbid', 'minshow', 'meanshow','min_per_show','max_per_show','showPNum',
                'accountid_show_max','accountid_bid_max','accountid_show_min','accountid_bid_min','accountid_show_mean','accountid_bid_mean',
                'shopid_show_max', 'shopid_bid_max', 'shopid_show_min', 'shopid_bid_min', 'shopid_show_mean','shopid_bid_mean',
                'shoptype_show_max', 'shoptype_bid_max', 'shoptype_show_min', 'shoptype_bid_min', 'shoptype_show_mean','shoptype_bid_mean',
                'industryid_show_max', 'industryid_bid_max', 'industryid_show_min', 'industryid_bid_min','industryid_show_mean', 'industryid_bid_mean',
                'push_long'
                ] + onehot_features
    onehot_features = []
    print(train.shape)
    print(train.info())
    preds = reg_model(train, test, train_label, model_type, onehot_features, label_features, features)



    if is_offline:
        #训练方式1
        # test_label = test['showNum']
        # smape, mse = eval_model(test_label, preds, 1)
        # print("score:", smape, mse)

        #训练方式2
        test['label'] = preds
        test2 = test2.merge(test[['origid','label']], how='left', on=['origid'])
        test2['preds'] = test2.apply(lambda x: x['bid']*x['label'], axis=1)
        test_label = test2['showNum']
        smape, mse = eval_model(test_label, test2['preds'], 1)
        print("score:", smape, mse)
    else:
        #训练方式2
        test['label'] = preds
        test2 = test2.merge(test[['origid', 'label']], how='left', on=['origid'])
        test2['preds'] = test2.apply(lambda x: round(x['bid'] * x['label'],4), axis=1)

        df = pd.DataFrame()
        df['id'] = test2['id']
        df['y'] = test2['preds']
        df.to_csv("./data/submissionA/model.csv", index=False, header=None)

'''
score: 0.9971306556591262 31544.725852910662
score: 0.99399116229978 31021.509996575594
score: 0.9309015162426666 24216.32767524884
score: 0.9208840269805185 21326.14291694565
score: 0.9003178452924103 18555.374841425044
score: 0.8980395111836456 15674.153269153869
score: 0.8969419554551985 15625.634356532168
#score: 0.8952229023152118 16121.30718707234 加入时间特征
score: 0.8934592290192702 15737.935059898207
score: 0.8930517245745755 15644.52259788667
score: 0.871598811032124 13924.310087500198
score: 0.8622524896124545 14840.757578139921
score: 0.8659885650423385 13198.04380563072 加入shopid
score: 0.863496130509283 12842.388884640062
score: 0.8698089182564406 12691.760714663831 industryid
score: 0.8675575150121594 12612.047287057945
score: 0.851918502291687 12826.707682704473  去除bid特征

score: 0.8679829003819252 13280.46852661964
'''
