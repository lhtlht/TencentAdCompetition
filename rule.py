import pandas as pd
import numpy as np
import sys
from feature_processing import *
from model import *


def mean_rule(x, show_bid_mean, requestDateWeek_mean_dict, size_mean_dict,shopid_mean_dict,shoptype_mean_dict,accountid_mean_dict):
    x_week = x['requestDateWeek']
    bid = x['bid']
    origidBidMean = x['origidBidMean']
    maxbid = x['maxbid']
    maxshow = x['maxshow']
    minbid = x['minbid']
    minshow = x['minshow']
    meanshow = x['meanshow']
    usergroup = x['usergroup']
    bid_list = [int(i) for i in x['bid_list'].split(',')]
    test_min_bid = min(bid_list)
    test_max_bid = max(bid_list)

    if origidBidMean >= 0.1:
        if maxshow <= minshow:
            return round(minshow / minbid * bid, 4)
        else:
            return round(origidBidMean,4)
    elif maxbid >= 0.1:
        if maxshow <= minshow:
            return round(minshow / minbid * bid, 4)
        else:
            slope = (maxshow - minshow) / (maxbid - minbid)
            bb = (bid - minbid)
            return round(minshow + bb * slope, 4)
    else:
        weights = []
        weights.append(requestDateWeek_mean_dict[x['requestDateWeek']])
        if x['size'] in size_mean_dict:
            weights.append(size_mean_dict[x['size']])
        if x['shopid'] in shopid_mean_dict:
            weights.append(shopid_mean_dict[x['shopid']])
        if x['shoptype'] in shoptype_mean_dict:
            weights.append(shoptype_mean_dict[x['shoptype']])
        if x['accountid'] in accountid_mean_dict:
            weights.append(accountid_mean_dict[x['accountid']])
        total_mean = np.mean(weights)
        if usergroup == 'all':
            total_mean = total_mean*1.2
        else:
            total_mean = total_mean * 0.8
        return round(total_mean * bid, 4)

def rule_model(train, test):
    import datetime
    train['requestDateWeek'] = train.apply(lambda x: datetime.datetime.strptime(x['requestDate'], '%Y-%m-%d').weekday(),axis=1)
    requestDateWeek_mean = train[['requestDateWeek', 'showNum', 'bid']].groupby('requestDateWeek').agg('mean').reset_index()
    requestDateWeek_mean['requestDateWeek_mean'] = requestDateWeek_mean['showNum'] / requestDateWeek_mean['bid']
    requestDateWeek_mean_dict = dict(zip(requestDateWeek_mean['requestDateWeek'], requestDateWeek_mean['requestDateWeek_mean']))

    size_mean = train[['size', 'showNum', 'bid']].groupby('size').agg('mean').reset_index()
    size_mean['size_mean'] = size_mean['showNum'] / size_mean['bid']
    size_mean_dict = dict(zip(size_mean['size'],size_mean['size_mean']))

    shopid_mean = train[['shopid', 'showNum', 'bid']].groupby('shopid').agg('mean').reset_index()
    shopid_mean['shopid_mean'] = shopid_mean['showNum'] / shopid_mean['bid']
    shopid_mean_dict = dict(zip(shopid_mean['shopid'], shopid_mean['shopid_mean']))

    shoptype_mean = train[['shoptype', 'showNum', 'bid']].groupby('shoptype').agg('mean').reset_index()
    shoptype_mean['shoptype_mean'] = shoptype_mean['showNum'] / shoptype_mean['bid']
    shoptype_mean_dict = dict(zip(shoptype_mean['shoptype'], shoptype_mean['shoptype_mean']))

    accountid_mean = train[['accountid', 'showNum', 'bid']].groupby('accountid').agg('mean').reset_index()
    accountid_mean['accountid_mean'] = accountid_mean['showNum'] / accountid_mean['bid']
    accountid_mean_dict = dict(zip(accountid_mean['accountid'], accountid_mean['accountid_mean']))



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
    sub['showPNum'] = sub.apply(lambda x: mean_rule(x, show_bid_mean, requestDateWeek_mean_dict, size_mean_dict,shopid_mean_dict,shoptype_mean_dict,accountid_mean_dict), axis=1)
    #sub['showPNum'] = sub.apply(lambda x: 0 if x['showPNum']<0 else x['showPNum'], axis=1)

    return sub

def redown_bids(bids):
    return ','.join([str(bid) for bid in bids])

if __name__ == "__main__":
    train = pd.read_csv(totalExposureLog_path3, encoding="utf-8")  # 02/16-03/19
    print(train.shape)
    ad_static_feature = pd.read_csv(ad_static_feature_path2, encoding="utf-8")
    origid_size = pd.read_csv(exposure_origid_size_path, encoding="utf-8")
    train = train.merge(ad_static_feature, how="left", on=["origid"])
    train.drop(['size'], axis=1, inplace=True)
    train = train.merge(origid_size, how="left", on=["origid"])
    test = pd.read_csv(test_sample_path2, encoding="utf-8")

    test_bid_list = test.groupby('origid').agg({'bid':redown_bids}).reset_index().rename(columns={'bid':'bid_list'})
    test = test.merge(test_bid_list, how='left', on='origid')



    sub = rule_model(train, test)
    print(sub.columns)
    sub[['id','showPNum']].to_csv("./data/submissionA/submission.csv", index=False, header=None)

    # model_ = pd.read_csv("./data/submissionA/submission_model.csv", encoding="utf-8", names=['id', 'model_show'])
    # sub = sub.merge(model_, how="left", on="id")
    # sub['showPNum'] = sub.apply(lambda x: round(x['showPNum']*0.9+x['model_show']*0.1,4) if x['maxbid']<0.1 else x['showPNum'], axis=1)
    # sub[['id', 'showPNum']].to_csv("./data/submissionA/0430_v1.csv", index=False, header=None)


    # rule_ = pd.read_csv("./data/submissionA/submission_rule.csv", encoding="utf-8", names=['id','rule_show'])
    # model_ = pd.read_csv("./data/submissionA/submission_model.csv", encoding="utf-8", names=['id','model_show'])
    # rule_model = pd.DataFrame()
    # rule_model = rule_.merge(model_, how="left", on="id")
    # rule_model['showNum'] = rule_model.apply(lambda x: round(x['rule_show']*0.9+x['model_show']*0.1,4), axis=1)
    # rule_model[['id','showNum']].to_csv("./data/submissionA/rule_model.csv", index=False, header=None)



