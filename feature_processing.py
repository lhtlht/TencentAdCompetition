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
totalExposureLog_path4 = PATH3 + "train4.csv"
totalExposureLog_path5 = PATH3 + "train5.csv"
user_data_path = PATH + "user_data"
user_data_path2 = PATH2 + "user_data.csv"
test_sample_path = PATH + 'test_sample.dat'
test_sample_path2 = PATH3 + 'test.csv'

exposure_origid_size_path = PATH2 + 'exposure_origid_size.csv'



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

    ad_static_feature.fillna(-1, inplace=True)
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

    ad_operation['ModifyTimeDate'] = ad_operation.apply(lambda x:
        str(x['createModifyTime'])[0:4]+'-'+str(x['createModifyTime'])[4:6]+'-'+str(x['createModifyTime'])[6:8] if x['createModifyTime'] != 0 else 0, axis=1)
    ad_operation_bid = ad_operation[(ad_operation['operationType'] == 2) & (ad_operation['modifyTag'] == 2)].rename(columns={'modifyDesc': 'bid'})
    ad_operation_usergroup = ad_operation[(ad_operation['operationType'] == 2) & (ad_operation['modifyTag'] == 3)].rename(columns={'modifyDesc': 'usergroup'})
    ad_operation_puttime = ad_operation[(ad_operation['operationType'] == 2) & (ad_operation['modifyTag'] == 4)].rename(columns={'modifyDesc': 'putTime'})

    ad_operation_bid2 = ad_operation[(ad_operation['operationType'] == 1) & (ad_operation['modifyTag'] == 2)].rename(columns={'modifyDesc': 'bid'})
    ad_operation_bid2 = ad_operation_bid2.drop_duplicates()
    ad_operation_bid2 = ad_operation_bid2.drop_duplicates(subset=['ModifyTimeDate', 'origid'], keep=False)

    ad_operation_usergroup2 = ad_operation[(ad_operation['operationType'] == 1) & (ad_operation['modifyTag'] == 3)].rename(columns={'modifyDesc': 'usergroup'})
    ad_operation_usergroup2 = ad_operation_usergroup2.drop_duplicates()
    ad_operation_usergroup2 = ad_operation_usergroup2.drop_duplicates(subset=['ModifyTimeDate', 'origid'], keep=False)

    ad_operation_puttime2 = ad_operation[(ad_operation['operationType'] == 1) & (ad_operation['modifyTag'] == 4)].rename(columns={'modifyDesc': 'putTime'})
    ad_operation_puttime2 = ad_operation_puttime2.drop_duplicates()
    ad_operation_puttime2 = ad_operation_puttime2.drop_duplicates(subset=['ModifyTimeDate', 'origid'], keep=False)

    ad_operation_re = ad_operation_bid
    ad_operation_re = ad_operation_re.merge(ad_operation_usergroup[['origid', 'usergroup']], how="left", on="origid")
    ad_operation_re = ad_operation_re.merge(ad_operation_puttime[['origid', 'putTime']], how="left", on="origid")

    ad_operation_re2 = ad_operation_bid2
    ad_operation_re2 = ad_operation_re2.merge(ad_operation_usergroup2[['origid', 'usergroup','ModifyTimeDate']],how='left', on=['origid','ModifyTimeDate'])

    ad_operation_re2 = ad_operation_re2.merge(ad_operation_puttime2[['origid', 'putTime','ModifyTimeDate']], how='left', on=['origid', 'ModifyTimeDate'])

    ad_operation_re = ad_operation_re.append(ad_operation_re2)
    ad_operation_re = ad_operation_re.append(ad_operation_usergroup2)
    ad_operation_re = ad_operation_re.append(ad_operation_puttime2)

    ad_operation_re.sort_values(by=['origid', 'createModifyTime'], axis=0, inplace=True)
    ad_operation_re.fillna(method='ffill', inplace=True)
    ad_operation_re = ad_operation_re.drop_duplicates(subset=['ModifyTimeDate', 'origid'], keep='first')
    ad_operation_re.to_csv(ad_operation_path2, index=False, encoding="utf-8")
    return ad_operation_re

def user_data_processing():
    user_data_columns = ["user_id", "age", "gender", "area", "status", "education", "consuptionAbility", "device", "work", "connectionType", "behavior"]
    user_data = pd.read_csv(user_data_path, encoding="utf-8", sep="\t", names=user_data_columns,header=None)
    user_data.to_csv(user_data_path2, index=False, encoding="utf-8")

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
def count_origid_bids(bids):
    return len(list(set(list(bids))))
def  totalExposureLog_processing2():
    test = pd.read_csv(test_sample_path2, encoding="utf-8")
    ad_operation = pd.read_csv(ad_operation_path2, encoding="utf-8")
    origid_operation = ad_operation.groupby('origid').size().reset_index()
    origids_test = test.groupby('origid').size().reset_index()
    origids = pd.concat([origid_operation, origids_test], axis=0)
    origids.drop_duplicates(subset=['origid'], inplace=True)
    print(origids.shape)



    totalExposureLog_columns = ["requestid", "requestTime", "positionid", "userid", "origid", "size", "bid", "pctr","quality_ecpm", "totalEcpm"]
    totalExposureLog_dtypes = {"requestid": "uint32", "requestTime": "uint32", "positionid": "int32","userid": "uint32", "origid": "int32",
                               "size": "int16",  "pctr": "float32", "quality_ecpm": "float32",
                               "totalEcpm": "float32"}
    totalExposureLog = pd.read_csv(totalExposureLog_path, encoding="utf-8", sep="\t",
                                   names=totalExposureLog_columns, header=None,
                                   dtype=totalExposureLog_dtypes)
    is_exposure_feature_processing = False
    if is_exposure_feature_processing:
        origid_size = totalExposureLog[['origid','size']].groupby(['origid','size']).size().reset_index()
        origid_size_drop_duplicates = origid_size.drop_duplicates(subset=['origid'], keep='first')
        print(origid_size.shape)
        print(origid_size_drop_duplicates.shape)
        origid_size_drop_duplicates.to_csv(exposure_origid_size_path, index=False, encoding="utf-8")
        #假设投放时间不变，计算投放时间
        #假设投放客群不变，计算投放客群大小
        #origid_put_time = totalExposureLog[['origid','requestTime']].groupby(['origid']).agg('max','min').reset_index()
        sys.exit(-1)
    totalExposureLog = totalExposureLog[["requestid","requestTime","positionid","origid","bid"]]
    #3.4G
    print(totalExposureLog.info())
    #删除重复曝光
    totalExposureLog.drop_duplicates(subset=['requestid','positionid','origid'],keep='first',inplace=True)
    print("删除重复曝光",totalExposureLog.shape)
    #缩小

    totalExposureLog = origids[['origid']].merge(totalExposureLog, how='left', on='origid')
    totalExposureLog.dropna(subset=['requestid'], inplace=True)
    totalExposureLog['requestDate'] = totalExposureLog.apply(lambda x: str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x['requestTime']))).split(' ')[0], axis=1)
    totalExposureLog.to_csv('bb.csv',index=False)
    print("格式化日期!")
    totalExposureLog = totalExposureLog[['requestDate', 'origid', 'bid']].groupby(['requestDate', 'origid']) \
        .agg(['size', 'max', 'min', 'mean', count_origid_bids]) \
        .reset_index()
    print("统计广告曝光次数",totalExposureLog.shape)
    totalExposureLog.columns = ['requestDate', 'origid', 'showNum', 'max_bid_inday', 'min_bid_inday', 'bid', 'bid_count']
    #去除一天bid变化的数据
    #totalExposureLog.drop_duplicates(subset=['requestDate', 'origid'],keep=False,inplace=True)
    #print("去除一天bid变化的数据",totalExposureLog.shape)
    totalExposureLog.to_csv(totalExposureLog_path4, index=False, encoding="utf-8")
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

def model_feature_processing(train, test):
    train['requestDateWeek'] = train.apply(lambda x: datetime.datetime.strptime(x['requestDate'], '%Y-%m-%d').weekday(), axis=1)
    train['diffdays'] = train.apply(lambda x:
        (datetime.datetime.strptime(x['requestDate'], '%Y-%m-%d')-datetime.datetime.strptime(x['createTimeDate'], '%Y-%m-%d')).days, axis=1)
    train['diffmonths'] = train.apply(lambda x:
        (datetime.datetime.strptime(x['requestDate'],'%Y-%m-%d') - datetime.datetime.strptime(x['createTimeDate'], '%Y-%m-%d')).days//30, axis=1)
    test['diffdays'] = test.apply(lambda x:
        (datetime.datetime.strptime(x['requestDate'],'%Y-%m-%d') - datetime.datetime.strptime(x['createTimeDate'], '%Y-%m-%d')).days, axis=1)
    test['diffmonths'] = test.apply(lambda x:
        (datetime.datetime.strptime(x['requestDate'], '%Y-%m-%d') - datetime.datetime.strptime(x['createTimeDate'], '%Y-%m-%d')).days // 30, axis=1)

    #去除行业id异常的数据
    train = train[train['industryid'].str.contains(',')==False]
    train['industryid'] = train['industryid'].astype(np.int32)
    #去除商品id异常的数据
    #train = train[train['shopid'].str.contains(',') == False]
    train['shopid'] = train['shopid'].astype(np.int32)
    #定位推送时间
    train['push_time'] = train.apply(lambda x: int(x['putTime'].split(',')[x['requestDateWeek']]) if x['putTime']!=0 else -1, axis=1)
    test['push_time'] = test.apply(lambda x: int(x['putTime'].split(',')[x['requestDateWeek']]) if x['putTime']!=0 else -1, axis=1)
    train['push_long'] = train.apply(lambda x: bin(x['push_time']).count('1') if x['push_time'] != -1 else -1, axis=1)
    test['push_long'] = test.apply(lambda x: bin(x['push_time']).count('1') if x['push_time'] != -1 else -1, axis=1)

    train['push_str'] = train.apply(lambda x: bin(x['push_time']).replace('0b',''), axis=1)
    test['push_str'] = test.apply(lambda x: bin(x['push_time']).replace('0b', ''), axis=1)

    train_push_hour = train.apply(lambda x: pd.Series(list('0'.zfill(48-len(x['push_str']))+x['push_str'])) if len(x['push_str'])<48 else pd.Series(list(x['push_str'])), axis=1)
    train_push_hour = pd.DataFrame(train_push_hour)
    train_push_hour.columns = ['hour'+str(i) for i in range(48)]
    train = pd.concat([train,train_push_hour], axis=1)
    train[['hour'+str(i) for i in range(48)]] = train[['hour'+str(i) for i in range(48)]].astype(np.int32)

    test_push_hour = test.apply(lambda x: pd.Series(list('0'.zfill(48-len(x['push_str']))+x['push_str'])) if len(x['push_str'])<48 else pd.Series(list(x['push_str'])), axis=1)
    test_push_hour = pd.DataFrame(test_push_hour)
    test_push_hour.columns = ['hour' + str(i) for i in range(48)]
    test = pd.concat([test, test_push_hour], axis=1)
    test[['hour' + str(i) for i in range(48)]] = test[['hour' + str(i) for i in range(48)]].astype(np.int32)
    #做曝光量的统计

    #针对test的处理
    #test['bid_count'] = np.ones(test.shape[0])
    #test['min_bid_inday'] = test['bid']
    #test['max_bid_inday'] = test['bid']


    train, test = train_sta(train, test)

    train['min_per_show'] = train['minshow']/train['minbid']
    train['max_per_show'] = train['maxshow'] / train['maxbid']

    test['min_per_show'] = test['minshow'] / test['minbid']
    test['max_per_show'] = test['maxshow'] / test['maxbid']

    train['bid'] = train['bid'].astype(np.int32)
    test['bid'] = test['bid'].astype(np.int32)
    return train, test

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
    show_max = train['showNum'].max()
    show_min = train['showNum'].min()
    #对周进行统计
    week_show_mean = train[['requestDateWeek', 'showNum']].groupby('requestDateWeek').mean().reset_index()
    week_bid_mean = train[['requestDateWeek', 'bid']].groupby('requestDateWeek').mean().reset_index()
    weeks = week_show_mean['showNum'] / week_bid_mean['bid']
    #对origid统计
    origid_max = train.groupby('origid').agg({'showNum': 'max', 'bid': 'max'}).reset_index()
    origid_max.columns = ['origid', 'origid_show_max', 'origid_bid_max']
    origid_min = train.groupby('origid').agg({'showNum': 'min', 'bid': 'min'}).reset_index()
    origid_min.columns = ['origid', 'origid_show_min', 'origid_bid_min']
    origid_mean = train.groupby('origid').agg({'showNum': 'mean', 'bid': 'mean'}).reset_index()
    origid_mean.columns = ['origid', 'origid_show_mean', 'origid_bid_mean']
    origid_sta = origid_max.merge(origid_min, how='left', on='origid')
    origid_sta = origid_sta.merge(origid_mean, how='left', on='origid')
    #对accountid特征进行统计
    accountid_max = train.groupby('accountid').agg({'showNum': 'max', 'bid': 'max'}).reset_index()
    accountid_max.columns = ['accountid', 'accountid_show_max', 'accountid_bid_max']
    accountid_min = train.groupby('accountid').agg({'showNum': 'min', 'bid': 'min'}).reset_index()
    accountid_min.columns = ['accountid', 'accountid_show_min', 'accountid_bid_min']
    accountid_mean = train.groupby('accountid').agg({'showNum': 'mean', 'bid': 'mean'}).reset_index()
    accountid_mean.columns = ['accountid', 'accountid_show_mean', 'accountid_bid_mean']
    accountid_sta = accountid_max.merge(accountid_min, how='left', on='accountid')
    accountid_sta = accountid_sta.merge(accountid_mean, how='left', on='accountid')
    # 对shopid特征进行统计
    shopid_max = train.groupby('shopid').agg({'showNum': 'max', 'bid': 'max'}).reset_index()
    shopid_max.columns = ['shopid', 'shopid_show_max', 'shopid_bid_max']
    shopid_min = train.groupby('shopid').agg({'showNum': 'min', 'bid': 'min'}).reset_index()
    shopid_min.columns = ['shopid', 'shopid_show_min', 'shopid_bid_min']
    shopid_mean = train.groupby('shopid').agg({'showNum': 'mean', 'bid': 'mean'}).reset_index()
    shopid_mean.columns = ['shopid', 'shopid_show_mean', 'shopid_bid_mean']
    shopid_sta = shopid_max.merge(shopid_min, how='left', on='shopid')
    shopid_sta = shopid_sta.merge(shopid_mean, how='left', on='shopid')
    # 对shoptype特征进行统计
    shoptype_max = train.groupby('shoptype').agg({'showNum': 'max', 'bid': 'max'}).reset_index()
    shoptype_max.columns = ['shoptype', 'shoptype_show_max', 'shoptype_bid_max']
    shoptype_min = train.groupby('shoptype').agg({'showNum': 'min', 'bid': 'min'}).reset_index()
    shoptype_min.columns = ['shoptype', 'shoptype_show_min', 'shoptype_bid_min']
    shoptype_mean = train.groupby('shoptype').agg({'showNum': 'mean', 'bid': 'mean'}).reset_index()
    shoptype_mean.columns = ['shoptype', 'shoptype_show_mean', 'shoptype_bid_mean']
    shoptype_sta = shoptype_max.merge(shoptype_min, how='left', on='shoptype')
    shoptype_sta = shoptype_sta.merge(shoptype_mean, how='left', on='shoptype')
    # 对industryid特征进行统计
    industryid_max = train.groupby('industryid').agg({'showNum': 'max', 'bid': 'max'}).reset_index()
    industryid_max.columns = ['industryid', 'industryid_show_max', 'industryid_bid_max']
    industryid_min = train.groupby('industryid').agg({'showNum': 'min', 'bid': 'min'}).reset_index()
    industryid_min.columns = ['industryid', 'industryid_show_min', 'industryid_bid_min']
    industryid_mean = train.groupby('industryid').agg({'showNum': 'mean', 'bid': 'mean'}).reset_index()
    industryid_mean.columns = ['industryid', 'industryid_show_mean', 'industryid_bid_mean']
    industryid_sta = industryid_max.merge(industryid_min, how='left', on='industryid')
    industryid_sta = industryid_sta.merge(industryid_mean, how='left', on='industryid')

    sub = test
    sub = sub.merge(origid_bid, how="left", on=['origid','bid'])
    sub = sub.merge(origidMax, how="left", on=['origid'])
    sub = sub.merge(origidMin, how="left", on=['origid'])
    sub = sub.merge(origidMean, how="left", on=['origid'])

    sub = sub.merge(origid_sta, how='left', on=['origid'])
    sub = sub.merge(accountid_sta, how='left', on=['accountid'])
    sub = sub.merge(shopid_sta, how='left', on=['shopid'])
    sub = sub.merge(shoptype_sta, how='left', on=['shoptype'])
    sub = sub.merge(industryid_sta, how='left', on=['industryid'])

    sub['origidBidMean'] = sub['origidBidMean'].fillna(show_max)
    sub['maxshow'] = sub['maxshow'].fillna(show_max)
    sub['minshow'] = sub['minshow'].fillna(show_min)
    sub['meanshow'] = sub['meanshow'].fillna(show_mean)

    sub['origid_show_max'] = sub['origid_show_max'].fillna(show_max)
    sub['origid_show_min'] = sub['origid_show_min'].fillna(show_min)
    sub['origid_show_mean'] = sub['origid_show_mean'].fillna(show_mean)
    sub['accountid_show_max'] = sub['accountid_show_max'].fillna(show_max)
    sub['accountid_show_min'] = sub['accountid_show_min'].fillna(show_min)
    sub['accountid_show_mean'] = sub['accountid_show_mean'].fillna(show_mean)
    sub['shopid_show_max'] = sub['shopid_show_max'].fillna(show_max)
    sub['shopid_show_min'] = sub['shopid_show_min'].fillna(show_min)
    sub['shopid_show_mean'] = sub['shopid_show_mean'].fillna(show_mean)
    sub['shoptype_show_max'] = sub['shoptype_show_max'].fillna(show_max)
    sub['shoptype_show_min'] = sub['shoptype_show_min'].fillna(show_min)
    sub['shoptype_show_mean'] = sub['shoptype_show_mean'].fillna(show_mean)
    sub['industryid_show_max'] = sub['industryid_show_max'].fillna(show_max)
    sub['industryid_show_min'] = sub['industryid_show_min'].fillna(show_min)
    sub['industryid_show_mean'] = sub['industryid_show_mean'].fillna(show_mean)

    sub['total_diff'] = sub['maxshow'] - sub['minshow']
    sub['origid_diff'] = sub['origid_show_max'] - sub['origid_show_min']
    sub['accountid_diff'] = sub['accountid_show_max'] - sub['accountid_show_min']
    sub['shopid_diff'] = sub['shopid_show_max'] - sub['shopid_show_min']
    sub['shoptype_diff'] = sub['shoptype_show_max'] - sub['shoptype_show_min']
    sub['industryid_diff'] = sub['industryid_show_max'] - sub['industryid_show_min']


    return sub

def last_sta(train, test, train_yesterday):
    show_mean = train['showNum'].mean()
    train.sort_values(by=['origid','requestDate'], ascending=[True,False],inplace=True)
    train = train.drop_duplicates(subset=['origid'], keep='first')
    train['last_show'] = train['showNum']

    yesterdays_mean = train[train['requestDate']==train_yesterday].mean()
    test = test.merge(train, how='left', on='origid')
    test['last_show'] = test['last_show'].fillna(show_mean)
    test['last_mean_show'] = pd.Series(np.zeros(test.shape[0]))+yesterdays_mean
    return test




def train_sta(train, test):
    """
    'maxbid', 'maxshow', 'minbid', 'minshow', 'meanshow', 'showPNum'
    """
    date_list = train.groupby('requestDate').size().index.tolist()
    train_rebuilt = pd.DataFrame()
    for date in date_list[1:]:
        print("sta show ",date)
        df_test = train[train['requestDate'] == date]
        #7天时间窗统计
        train_date_pre = (datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
        df_train = train[(train['requestDate']<date) &  (train['requestDate']>=train_date_pre)]
        df = rule_model(df_train, df_test)
        #历史统计
        df_train_history = train[train['requestDate'] < date]
        df_history = rule_model(df_train_history, df_test)
        df_history = df_history.rename(columns={'meanshow':'meanshow_history',
                                                'maxshow':'maxshow_history',
                                                'minshow':'minshow_history',
                                                'origid_show_mean': 'origid_show_mean_history',
                                                'accountid_show_mean': 'accountid_show_mean_history',
                                                'shopid_show_mean': 'shopid_show_mean_history',
                                                'shoptype_show_mean': 'shoptype_show_mean_history',
                                                'industryid_show_mean': 'industryid_show_mean_history',
                                                })
        #最近统计
        train_yesterday = (datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        df_last = last_sta(df_train_history, df_test, train_yesterday)

        df = pd.concat([df,
                        df_history[['meanshow_history','maxshow_history','minshow_history',
                                    'origid_show_mean_history', 'accountid_show_mean_history',
                                    'shopid_show_mean_history',
                                    'shoptype_show_mean_history', 'industryid_show_mean_history',
                                    ]],
                        df_last[['last_show','last_mean_show']]], axis=1)
        train_rebuilt = train_rebuilt.append(df)

    df_train = train[train['requestDate'] >= '2019-03-12']
    #7天窗口
    test_window = rule_model(df_train, test)
    #历史
    test_history = rule_model(train, test)
    test_history = test_history.rename(columns={'meanshow': 'meanshow_history',
                                            'maxshow': 'maxshow_history',
                                            'minshow': 'minshow_history',
                                               'origid_show_mean':'origid_show_mean_history',
                                               'accountid_show_mean':'accountid_show_mean_history',
                                               'shopid_show_mean':'shopid_show_mean_history',
                                               'shoptype_show_mean':'shoptype_show_mean_history',
                                               'industryid_show_mean':'industryid_show_mean_history',
                                               })
    #最近统计
    train_yesterday = '2019-03-19'
    test_last = last_sta(train, test, train_yesterday)
    test_rebuilt = pd.concat([test_window,
                              test_history[['meanshow_history', 'maxshow_history', 'minshow_history',
                                            'origid_show_mean_history', 'accountid_show_mean_history',
                                            'shopid_show_mean_history',
                                            'shoptype_show_mean_history', 'industryid_show_mean_history',
                                            ]],
                              test_last[['last_show','last_mean_show']]], axis=1)
    return train_rebuilt, test_rebuilt



if __name__ == "__main__":
    #第一步处理
    #ad_static_feature_processing()
    #ad_operation_processing()
    #totalExposureLog_processing()
    #user_data_processing()
    #test_sample_processing()

    #额外的曝光日志处理
    totalExposureLog_processing2()


