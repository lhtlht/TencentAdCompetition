import pandas as pd
import numpy as np
import pickle
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

def modify_time_date_list(ModifyTimeDates):
    return list(ModifyTimeDates)
if __name__ == "__main__":
    is_offline = False
    is_load_data = True
    if is_load_data:
        train, test, ad_operation, ad_static_feature, origid_size = load_data()
        #训练数据的整理
        origid_operation = ad_operation.groupby('origid').size().reset_index()
        origids_test = test.groupby('origid').size().reset_index()
        origids = pd.concat([origid_operation, origids_test], axis=0)
        train = origids[['origid']].merge(train, how='left', on='origid')
        train.dropna(subset=['requestDate'], inplace=True) #去除某些设置，然而没有曝光的广告
        train = train.merge(ad_static_feature, how="left", on=["origid"])
        train.drop(['size'], axis=1, inplace=True)
        train = train.merge(origid_size, how="left", on=["origid"])


        train.dropna(subset=['createTimeDate'], inplace=True)
        ad_operation = ad_operation.merge(ad_static_feature[['origid','createTimeDate']], how='left', on=['origid'])
        ad_operation['ModifyTimeDate'] = ad_operation.apply(lambda x: x['createTimeDate'] if x['operationType']==2 else x['ModifyTimeDate'], axis=1)
        ad_operation.drop(['createTimeDate'], axis=1, inplace=True)
        ad_operation.drop(['createModifyTime'], axis=1, inplace=True)
        ad_operation.drop(['modifyTag'], axis=1, inplace=True)
        ad_operation.drop(['operationType'], axis=1, inplace=True)

        f_train = open(PATH3+'train.pkl', 'wb')
        pickle.dump(train, f_train)
        f_train.close()

        f_test = open(PATH3 + 'test.pkl', 'wb')
        pickle.dump(test, f_test)
        f_test.close()
    else:
        f_train = open(PATH3 + 'train.pkl', 'rb')
        train = pickle.load(f_train)
        f_train.close()

        f_test = open(PATH3 + 'test.pkl', 'rb')
        test = pickle.load(f_test)
        f_test.close()
        #ad_operation.fillna(0, inplace=True)
        #train = train.merge(ad_operation, how="left", on=["origid"])
        #origid_mod = ad_operation.groupby('origid').agg({'ModifyTimeDate':modify_time_date_list}).reset_index().rename(columns={'ModifyTimeDate':'ModifyTimeDates'}).to_dict(orient='dict')
        #train['is_drop'] = train.apply(lambda x: 1 if x['requestDate']>=x['ModifyTimeDate'] else 0, axis=1)
        #train = train[train['is_drop']==1]
        #train.sort_values(by=['origid', 'createModifyTime'], axis=0, inplace=True)
        #train.fillna(method='bfill', inplace=True)
    print(train.shape)
    print(train.columns)
    print(test.shape)
    print(test.columns)

    #特征处理
    """
     'maxbid', 'maxshow', 'minbid', 'minshow', 'meanshow', 'showPNum'
     
    """


    train.fillna(0, inplace=True)

    train, test = model_feature_processing(train, test)
    train = train[train['requestDate'] >= '2019-03-01']
    if is_offline:
        test = train[train['requestDate'] == '2019-03-19']
        train = train[train['requestDate'] != '2019-03-19']

    train_label = train['showNum']
    test2 = test.copy()
    test = test.drop_duplicates(subset=['origid','requestDate'], keep='first')
    #模型训练

    model_type = 'lgb'
    label_features = ['accountid', 'shoptype', 'origid', 'industryid',
                      'createTimeDay', 'createTimeWeek', 'createTimeYear', 'createTimeMonth', 'createTimeHour']
    onehot_features = ['accountid', 'shoptype', 'origid', 'industryid',
                      'requestDateWeek','createTimeWeek',
                       'diffdays','diffmonths']
    features = [ 'size', 'maxshow', 'minshow', 'meanshow',
                'accountid_show_max','accountid_show_min','accountid_show_mean',
                'shopid_show_max', 'shopid_show_min', 'shopid_show_mean',
                'shoptype_show_max', 'shoptype_show_min', 'shoptype_show_mean',
                'industryid_show_max', 'industryid_show_min','industryid_show_mean',
                ]
    #onehot_features = []
    print(train.shape)
    print(train.info())
    preds = reg_model(train, test, train_label, model_type, onehot_features, label_features, features)

    if is_offline:
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
        test = test.rename(columns={'bid':'pre_bid'})
        print(test.columns)
        test2 = test2.merge(test[['origid','label','pre_bid']], how='left', on=['origid'])
        test2['preds'] = test2.apply(lambda x: x['label']+(x['bid']-x['pre_bid'])*0.0001, axis=1)


        test2[['origid','bid','preds','pre_bid']].to_csv("./data/submissionA/model_test.csv", index=False)
        df = pd.DataFrame()
        df['id'] = test2['id']
        df['y'] = test2['preds']
        df['y'] = df.apply(lambda x: round(x['y'],4), axis=1)
        df.to_csv("./data/submissionA/model.csv", index=False, header=None)

'''

'''
