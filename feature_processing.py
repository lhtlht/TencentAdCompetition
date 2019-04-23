import time
import datetime
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)
PATH="./data/testA/"
PATH2="./data/testA_/"
PATH3 = "./data/testA_tmp/"
ad_static_feature_path = PATH + "ad_static_feature.out"
ad_static_feature_path2 = PATH2 + "ad_static_feature.csv"
ad_operation_path = PATH + "ad_operation.dat"
ad_operation_path2 = PATH2 + "ad_operation.csv"
totalExposureLog_path = PATH + "totalExposureLog.out"
totalExposureLog_path2 = PATH3 + "train.csv"
totalExposureLog_path3 = PATH3 + "train_processing.csv"
test_sample_path = PATH + 'test_sample.dat'
test_sample_path2 = PATH3 + 'test.csv'

def ad_static_feature_processing():
    """
       rows:735911
       origid:735911
       size:509252,缺失226659
       shopid:缺失157214
       industryid：有逗号分隔
    """
    ad_static_feature_columns = ["origid", "createTime", "accountid", "shopid", "shoptype", "industryid", "size"]
    ad_static_feature = pd.read_csv(ad_static_feature_path, encoding="utf-8", sep="\t", names=ad_static_feature_columns, header=None)

    ad_static_feature.fillna(int(ad_static_feature['size'].mode()), inplace=True)
    ad_static_feature['size'] = ad_static_feature.apply(lambda x: -1 if x['size'] == '17,57' else int(x['size']),axis=1)
    ad_static_feature['createTime'] = ad_static_feature.apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x['createTime'])), axis=1)
    ad_static_feature['createTimeDate'] = ad_static_feature.apply(lambda x: x['createTime'].split(' ')[0], axis=1)
    ad_static_feature['createTimeYear'] = ad_static_feature.apply(lambda x: x['createTime'].split(' ')[0].split('-')[0],axis=1)
    ad_static_feature['createTimeMonth'] = ad_static_feature.apply(lambda x: x['createTime'].split(' ')[0].split('-')[1], axis=1)
    ad_static_feature['createTimeDay'] = ad_static_feature.apply(lambda x: x['createTime'].split(' ')[0].split('-')[2],axis=1)
    ad_static_feature['createTimeHour'] = ad_static_feature.apply(lambda x: x['createTime'].split(' ')[1].split(':')[0],axis=1)
    ad_static_feature['createTimeWeek'] = ad_static_feature.apply(lambda x: datetime.datetime.strptime(x['createTimeDate'], '%Y-%m-%d').weekday(),axis=1)
    ad_static_feature.to_csv(ad_static_feature_path2, index=False, encoding="utf-8")
    return ad_static_feature

def ad_operation_processing():
    """
    rows:760866
    origid：38843
    operationType:
        1    648903  修改  基本上都有时间
        2    111963  新建  大部分为0,并且设置出价 2
    modifyTag：
        1    500802  广告状态
        2    167298  出价
        3     47411  人群定向
        4     45355  广告时段设置
    :return:
    """
    ad_operation_columns = ["origid", "createModifyTime", "operationType", "modifyTag", "modifyDesc"]
    ad_operation = pd.read_csv(PATH + "ad_operation.dat", encoding="utf-8", sep="\t", names=ad_operation_columns,header=None)

    ad_operation['ModifyTimeDate'] = ad_operation.apply(lambda x: str(x['createModifyTime'])[0:8] if x['createModifyTime'] != 0 else 0, axis=1)
    ad_operation_bid = ad_operation[(ad_operation['operationType'] == 2) & (ad_operation['modifyTag'] == 2)].rename(columns={'modifyDesc': 'bid'})
    ad_operation_usergroup = ad_operation[(ad_operation['operationType'] == 2) & (ad_operation['modifyTag'] == 3)].rename(columns={'modifyDesc': 'usergroup'})
    ad_operation_puttime = ad_operation[(ad_operation['operationType'] == 2) & (ad_operation['modifyTag'] == 4)].rename(columns={'modifyDesc': 'putTime'})

    ad_operation_bid2 = ad_operation[(ad_operation['operationType'] == 1) & (ad_operation['modifyTag'] == 2)].rename(columns={'modifyDesc': 'bid'})
    ad_operation_usergroup2 = ad_operation[(ad_operation['operationType'] == 1) & (ad_operation['modifyTag'] == 3)].rename(columns={'modifyDesc': 'usergroup'})
    ad_operation_puttime2 = ad_operation[(ad_operation['operationType'] == 1) & (ad_operation['modifyTag'] == 4)].rename(columns={'modifyDesc': 'putTime'})

    ad_operation_re = ad_operation_bid
    ad_operation_re = ad_operation_re.merge(ad_operation_usergroup[['origid', 'usergroup']], how="left", on="origid")
    ad_operation_re = ad_operation_re.merge(ad_operation_puttime[['origid', 'putTime']], how="left", on="origid")

    ad_operation_re = ad_operation_re.append(ad_operation_bid2)
    ad_operation_re = ad_operation_re.append(ad_operation_usergroup2)
    ad_operation_re = ad_operation_re.append(ad_operation_puttime2)

    ad_operation_re.sort_values(by=['origid', 'createModifyTime'], axis=0, inplace=True)
    ad_operation_re.fillna(method='bfill', inplace=True)

    ad_operation_re.to_csv(ad_operation_path2, index=False, encoding="utf-8")
    return ad_operation_re

def  totalExposureLog_processing():
    totalExposureLog_columns = ["requestid", "requestTime", "positionid", "userid", "origid", "size", "bid", "pctr","quality_ecpm", "totalEcpm"]
    totalExposureLog_dtypes = {"requestid": "uint32", "requestTime": "uint32", "positionid": "int32","userid": "uint32", "origid": "int32",
                               "size": "int16", "bid": "int16", "pctr": "float32", "quality_ecpm": "float32",
                               "totalEcpm": "float32"}
    totalExposureLog = pd.read_csv(totalExposureLog_path, encoding="utf-8", sep="\t",
                                   names=totalExposureLog_columns, header=None,
                                   dtype=totalExposureLog_dtypes,
                                   iterator=True)
    #3.4G
    totalExposureLogData = pd.DataFrame()
    chunk_num = 1
    t1 = time.time()
    while True:
        try:
            print(f'chunk num : {chunk_num} is processing......')
            totalExposureLogChunk = totalExposureLog.get_chunk(20000000)
            totalExposureLogChunk['requestDate'] = totalExposureLogChunk.apply(
                lambda x: str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x['requestTime']))).split(' ')[0], axis=1)
            df = totalExposureLogChunk[['requestDate', 'origid', 'bid']].groupby(['requestDate', 'origid', 'bid'])\
                .size() \
                .reset_index() \
                .rename(columns={0: 'showNum'})
            totalExposureLogData = totalExposureLogData.append(df)
            print(totalExposureLogData.shape)
            chunk_num = chunk_num + 1
        except StopIteration:
            print("Iteration is stopped.")
            break

    print(f'chunknums:{chunk_num},user time:',time.time()-t1)
    print(totalExposureLogData.info())
    totalExposureLogDataProcess = totalExposureLogData.groupby(['requestDate', 'origid', 'bid']).sum().reset_index()
    totalExposureLogDataProcess.to_csv(totalExposureLog_path2, index=False, encoding="utf-8")
def  totalExposureLog_processing2():
    totalExposureLog_columns = ["requestid", "requestTime", "positionid", "userid", "origid", "size", "bid", "pctr","quality_ecpm", "totalEcpm"]
    totalExposureLog_dtypes = {"requestid": "uint32", "requestTime": "uint32", "positionid": "int32","userid": "uint32", "origid": "int32",
                               "size": "int16", "bid": "int16", "pctr": "float32", "quality_ecpm": "float32",
                               "totalEcpm": "float32"}
    totalExposureLog = pd.read_csv(totalExposureLog_path, encoding="utf-8", sep="\t",
                                   names=totalExposureLog_columns, header=None,
                                   dtype=totalExposureLog_dtypes)
    totalExposureLog = totalExposureLog[["requestid","requestTime","positionid","origid","bid"]]
    #3.4G
    print(totalExposureLog.info())
    #删除重复曝光
    totalExposureLog.drop_duplicates(subset=['requestid','positionid','origid'],keep='first',inplace=True)
    print("删除重复曝光",totalExposureLog.info())
    totalExposureLog['requestDate'] = totalExposureLog.apply(lambda x: str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x['requestTime']))).split(' ')[0], axis=1)
    totalExposureLog = totalExposureLog[['requestDate', 'origid', 'bid']].groupby(['requestDate', 'origid', 'bid']) \
        .size() \
        .reset_index() \
        .rename(columns={0: 'showNum'})
    print("统计广告曝光次数",totalExposureLog.shape)
    #去除一天bid变化的数据
    totalExposureLog.drop_duplicates(subset=['requestDate', 'origid'],keep=False,inplace=True)
    print("去除一天bid变化的数据",totalExposureLog.shape)
    totalExposureLog.to_csv(totalExposureLog_path3, index=False, encoding="utf-8")
def preDate(date, pre):
    if date >= '2019-03-20':
        return (datetime.datetime.strptime(date, '%Y-%m-%d') + datetime.timedelta(days=pre)).strftime("%Y-%m-%d")
    else:
        return '2019-03-21'

def test_sample_processing():
    test_sample_columns = ["id", "origid", "createTime", "size", "industryid", "shoptype", "shopid", "accountid","putTime", "usergroup", "bid"]
    test_sample = pd.read_csv(test_sample_path, encoding="utf-8", sep="\t", names=test_sample_columns, header=None)
    test_sample['createTime'] = test_sample.apply(lambda x: str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x['createTime']))), axis=1)
    test_sample['createTimeDate'] = test_sample.apply(lambda x: x['createTime'].split(' ')[0], axis=1)
    test_sample['createTimeYear'] = test_sample.apply(lambda x: x['createTime'].split(' ')[0].split('-')[0], axis=1)
    test_sample['createTimeMonth'] = test_sample.apply(lambda x: x['createTime'].split(' ')[0].split('-')[1], axis=1)
    test_sample['createTimeDay'] = test_sample.apply(lambda x: x['createTime'].split(' ')[0].split('-')[2], axis=1)
    test_sample['createTimeHour'] = test_sample.apply(lambda x: x['createTime'].split(' ')[1].split(':')[0], axis=1)
    test_sample['createTimeWeek'] = test_sample.apply(lambda x: datetime.datetime.strptime(x['createTimeDate'], '%Y-%m-%d').weekday(), axis=1)

    test_sample['requestDate'] = test_sample.apply(lambda x: preDate(x['createTimeDate'],1), axis=1)
    test_sample['requestDateWeek'] = test_sample.apply(lambda x: datetime.datetime.strptime(x['requestDate'], '%Y-%m-%d').weekday(), axis=1)
    test_sample.to_csv(test_sample_path2, index=False, encoding="utf-8")
    return test_sample

def model_feature_processing(train, test, train_label):
    train['requestDateWeek'] = train.apply(lambda x: datetime.datetime.strptime(x['requestDate'], '%Y-%m-%d').weekday(), axis=1)
    #做曝光量的统计
    train, test, train_label = train_sta(train, test, train_label)
    return train, test, train_label

def mean_rule(x, show_bid_mean):
    bid = x['bid']
    origidBidMean = x['origidBidMean']
    maxbid = x['maxbid']
    maxshow = x['maxshow']
    minbid = x['minbid']
    minshow = x['minshow']
    meanshow = x['meanshow']
    if origidBidMean >= 0.1:
        return origidBidMean
    elif maxbid >= 0.1:
        if maxshow <= minshow:
            return round(minshow / (minbid+0.1) * bid, 4)
        else:
            slope = (maxshow - minshow) / (maxbid - minbid)
            bb = bid - minbid
            return round(minbid + bb * slope, 4)
    else:
        return round(show_bid_mean * bid, 4)

def rule_model(train, test):

    origid_bid = train[['origid','bid','showNum']].groupby(['origid','bid']).mean().reset_index().rename(columns={'showNum':'origidBidMean'})
    origidMax = train[['origid','bid']].groupby(['origid']).max().reset_index()
    origidMin = train[['origid','bid']].groupby(['origid']).min().reset_index()

    origidMax = origidMax.merge(train[['origid','bid','showNum']], how='left', on=['origid','bid'])
    origidMin = origidMin.merge(train[['origid','bid','showNum']], how='left', on=['origid','bid'])
    origidMax = origidMax.groupby(['origid','bid']).mean().reset_index().rename(columns={'bid':'maxbid', 'showNum':'maxshow'})
    origidMin = origidMin.groupby(['origid','bid']).mean().reset_index().rename(columns={'bid':'minbid', 'showNum':'minshow'})
    origidMean = train[['origid','showNum']].groupby(['origid']).mean().reset_index().rename(columns={'showNum':'meanshow'})
    show_mean = train['showNum'].mean()
    bid_mean = train['bid'].mean()
    show_bid_mean = show_mean/bid_mean

    sub = test
    sub = sub.merge(origid_bid, how="left", on=['origid','bid'])


    sub = sub.merge(origidMax, how="left", on=['origid'])
    sub = sub.merge(origidMin, how="left", on=['origid'])
    sub = sub.merge(origidMean, how="left", on=['origid'])
    sub.fillna(0,inplace=True)
    sub['showPNum'] = sub.apply(lambda x: mean_rule(x, show_bid_mean), axis=1)


    return sub

def train_sta(train, test, train_label):
    """
    'maxbid', 'maxshow', 'minbid', 'minshow', 'meanshow', 'showPNum'
    """
    date_list = train.groupby('requestDate').size().index.tolist()
    train_rebuilt = pd.DataFrame()
    train['showNum'] = train_label
    for date in date_list[1:]:
        print("sta show ",date)
        train_date_pre = (datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
        #df_train = train[(train['requestDate']<date) and  (train['requestDate']>=train_date_pre)]
        df_train = train[train['requestDate'] < date]
        df_test = train[train['requestDate']==date]

        df = rule_model(df_train, df_test)
        train_rebuilt = train_rebuilt.append(df)
    train_label = train_rebuilt['showNum']
    train_rebuilt = train_rebuilt.drop(columns=['showNum'])
    test_rebuilt = rule_model(train, test)
    return train_rebuilt, test_rebuilt, train_label



if __name__ == "__main__":
    #第一步处理
    #ad_static_feature_processing()
    #ad_operation_processing()
    #totalExposureLog_processing()
    #test_sample_processing()

    #额外的曝光日志处理
    totalExposureLog_processing2()

    #第二步处理
    #展现的均值回溯

