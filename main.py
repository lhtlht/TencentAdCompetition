import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def join_puttime(x):
    modify_time_date_list = x['modify_time_date_list']

    ModifyTimeDate = x['ModifyTimeDate']
    requestDate = x['requestDate']
    if modify_time_date_list != modify_time_date_list:
        return 2
    right_date = modify_time_date_list[len(modify_time_date_list)-1]
    right_status = 0
    for i in range(len(modify_time_date_list)-1):
        if requestDate>=modify_time_date_list[i] and requestDate<modify_time_date_list[i+1]:
            right_date = modify_time_date_list[i]
            break
    if right_date == ModifyTimeDate:
        right_status = 1
    else:
        right_status = 0

    return right_status
if __name__ == "__main__":
    is_offline = True
    is_load_data = False
    if is_load_data:
        train, test, ad_operation, ad_static_feature, origid_size = load_data()
        # 去除异常值
        #train = train[(train['showNum'] < 100) & (train['bid'] < 1000)]
        #训练数据的整理
        #origid_operation = ad_operation.groupby('origid').size().reset_index()
        test_tmp = test.copy()
        test_tmp = test_tmp.rename(columns={'putTime':'putTimeTest','usergroup':'usergroupTest'})
        #origids_test = test_tmp.groupby('origid').size().reset_index()
        #origids = pd.concat([origid_operation, origids_test], axis=0)
        #train = origids[['origid']].merge(train, how='left', on='origid')
        #train.dropna(subset=['requestDate'], inplace=True) #去除某些设置，然而没有曝光的广告
        train = train.merge(ad_static_feature, how="left", on=["origid"])
        train.drop(['size'], axis=1, inplace=True)
        train = train.merge(origid_size, how="left", on=["origid"])
        train.dropna(subset=['createTimeDate'], inplace=True)

        ad_operation = ad_operation.merge(ad_static_feature[['origid','createTimeDate']], how='left', on=['origid'])
        ad_operation['ModifyTimeDate'] = ad_operation.apply(lambda x: x['createTimeDate'] if x['operationType']==2 else x['ModifyTimeDate'], axis=1)
        ad_operation = ad_operation.drop_duplicates(subset=['ModifyTimeDate', 'origid'], keep='first')
        ad_operation.drop(['createTimeDate'], axis=1, inplace=True)
        ad_operation.drop(['createModifyTime'], axis=1, inplace=True)
        ad_operation.drop(['modifyTag'], axis=1, inplace=True)
        ad_operation.drop(['operationType'], axis=1, inplace=True)
        ad_operation.drop(['bid'], axis=1, inplace=True)
        #对投放时间、客群的拼接
        origid_timelist = ad_operation.groupby('origid').agg({'ModifyTimeDate':modify_time_date_list}).reset_index()
        origid_timelist = origid_timelist.rename(columns={'ModifyTimeDate':'modify_time_date_list'})
        ad_operation = ad_operation.merge(origid_timelist, how='left', on='origid')
        print(train.shape)
        train = train.merge(ad_operation, how='left', on='origid')
        train['is_drop'] = train.apply(lambda x: join_puttime(x), axis=1)
        print(train.groupby('is_drop').size())
        test_tmp = test_tmp.drop_duplicates(subset=['origid'])
        train = train.merge(test_tmp[['origid','usergroupTest','putTimeTest']], how='left', on=['origid'])
        train['putTime'] = train.apply(lambda x: x['putTimeTest'] if x['is_drop']==2 else x['putTime'],axis=1)
        train['usergroup'] = train.apply(lambda x: x['usergroupTest'] if x['is_drop'] == 2 else x['usergroup'], axis=1)
        train = train[train['is_drop']!=0]
        train.drop(['is_drop'], axis=1, inplace=True)
        train.drop(['modify_time_date_list'], axis=1, inplace=True)
        print(train.shape)
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

    print(train.shape)
    print(train.columns)
    print(test.shape)
    print(test.columns)

    #特征处理
    train.fillna(0, inplace=True)

    train, test = model_feature_processing(train, test)
    train = train[train['requestDate'] >= '2019-03-01']
    if is_offline:
        test = train[train['requestDate'] == '2019-03-19']
        train = train[train['requestDate'] != '2019-03-19']
    else:
        test2 = test.copy()
        test = test.drop_duplicates(subset=['origid','requestDate'], keep='first')
    train_label = train['showNum']
    #模型训练

    model_type = 'lgb'
    put_hour_feature = ['hour'+str(i) for i in range(48)]

    label_features = ['accountid', 'shoptype', 'origid', 'industryid','shopid',
                      'createTimeDay', 'createTimeWeek', 'createTimeYear', 'createTimeMonth', 'createTimeHour']
    onehot_features = ['accountid', 'shoptype', 'origid', 'industryid','shopid',
                      'requestDateWeek','createTimeWeek',
                       'diffdays','diffmonths']
    features = [ 'size', 'maxshow', 'minshow', 'meanshow','origidBidMean','total_diff',
                'origid_show_mean',

                ] + onehot_features
    onehot_features = []
    train[label_features] = train[label_features].astype(int)
    test[label_features] = test[label_features].astype(int)
    label_features = []

    print(train.shape)
    print(train.info())
    print(test.info())
    preds = reg_model(train, test, train_label, model_type, onehot_features, label_features, features)

    if is_offline:
        #训练方式2
        test['label'] = preds
        test_label = test['showNum']
        smape, mse = eval_model(test_label, test['label'], 1)
        print("score:", smape, mse)
    else:
        #训练方式2
        test['label'] = preds
        test['sl'] = test['label']/test['bid']
        test = test.rename(columns={'bid':'pre_bid'})
        print(test.columns)
        test2 = test2.merge(test[['origid','label','pre_bid','sl']], how='left', on=['origid'])
        test2['preds'] = test2.apply(lambda x: x['label']+(x['bid']-x['pre_bid'])*0.0001, axis=1)


        test2[['origid','bid','preds','pre_bid']].to_csv("./data/submissionA/model_test.csv", index=False)
        df = pd.DataFrame()
        df['id'] = test2['id']
        df['y'] = test2['preds']
        df['y'] = df.apply(lambda x: round(x['y'],4), axis=1)
        print(df['y'].mean())
        df.to_csv("./data/submissionA/submission.csv", index=False, header=None)

'''
score: 0.7341809492043748 22533.031697817303
score: 0.7337858462047834 23270.84441165036
score: 0.723468839768221 23477.177423585505
score: 0.7265117959409166 21906.77865294984
score: 0.7215252196709249 23027.287463769775

score: 0.7190696074609102 22391.06278183768
score: 0.7183270629199415 21902.71653019669

score: 0.7145889048509767 22064.342145466104

31.031380857565303
30.930490758994576

score: 0.7568063510699478 24020.611940457777
score: 0.7603950244457541 23064.22292584241
score: 0.7358247706238431 23402.50772997647
score: 0.7097136847975927 26390.197786279223
score: 0.6999918732066851 25515.319187750723
score: 0.6968026069580829 25124.520068146852
score: 0.7211974878072482 23026.883239835137
'''
