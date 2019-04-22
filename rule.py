import pandas as pd
import numpy as np
import sys
from feature_processing import *
from model import *
train_dtypes = {'origid':'uint32', 'bid':'int16'}
train = pd.read_csv(totalExposureLog_path2, encoding="utf-8", dtype=train_dtypes) #02/16-03/19
test = pd.read_csv(test_sample_path2, encoding="utf-8")

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

sub = test[['id','origid','bid']]
sub = sub.merge(origid_bid, how="left", on=['origid','bid'])
sub = sub.merge(origidMax, how="left", on=['origid'])
sub = sub.merge(origidMin, how="left", on=['origid'])
sub = sub.merge(origidMean, how="left", on=['origid'])
sub.fillna(0,inplace=True)


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
            return round(minshow / minbid * bid, 4)
        else:
            slope = (maxshow - minshow) / (maxbid - minbid)
            bb = bid - minbid
            return round(minbid + bb * slope, 4)
    else:
        return round(show_bid_mean * bid, 4)


sub['showNum'] = sub.apply(lambda x: mean_rule(x, show_bid_mean), axis=1)
sub[['id', 'showNum']].to_csv("./data/submissionA/0422_v2.csv", index=False, header=None)
sub.head(20)