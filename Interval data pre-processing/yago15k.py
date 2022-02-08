# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import os, pdb

os.chdir("D:/MU/05.HWS2021/Thesis/tmp-kge/data")


# data loading
# 110441
train = pd.read_table('yago15k/yago15k_train.txt', sep = '\t', header = None,
                        names = ['S','P','O','M','T'])
# 13815
valid = pd.read_table('yago15k/yago15k_valid.txt', sep = '\t', header = None,
                        names = ['S','P','O','M','T'])
# 13800
test = pd.read_table('yago15k/yago15k_test.txt', sep = '\t', header = None,
                        names = ['S','P','O','M','T'])

# 138056
alldata = pd.concat([train,valid,test])

# only 35 examples have month or day
# In this study, only year is considered
len(alldata[(alldata["T"].str[-5:] != "##-##") & ~(alldata["T"].isna())])


# delete two records since these records have modifier but do not have timestamp
# <India>  <participatedIn>  <Korean_War>  <occursUntil>  NaN
# <India>  <participatedIn>  <Korean_War>  <occursSince>  NaN 
train[~(train["M"].isna()) & (train["T"].isna())]

train = train[~(~(train["M"].isna()) & (train["T"].isna()))]


YEARMAX = 2020
YEARMIN = 0

# create year2id 
year = train['T']
years = year[~(year.isna())].str.split('-', expand=True)[[0]].astype('int').rename({0:'year'}, axis=1)
year_freq = years['year'].value_counts().sort_index()

year_class=[]
count=0
for idx in year_freq.index:
    count += year_freq[idx]
    if count>=300:
        year_class.append(idx)
        count=0
year_class[-1]=YEARMAX

year2id = dict()
prev_year = YEARMIN
i = 0
for i, yr in enumerate(year_class): 
    year2id[(prev_year, yr)] = i
    prev_year = yr + 1


### Method 1: All year information generaion 
def preprcessing_m1(dataset):
    
    # since data
    _since = dataset[dataset['M'] == "<occursSince>"].drop('M', axis=1)
    # until data
    _until = dataset[dataset['M'] == "<occursUntil>"].drop('M', axis=1)
    # non-temporal data
    _nt = dataset[dataset['M'].isna()].rename(columns={"M":"st", "T":"ed"})
    # merge since and until to one records with start and end time
    _temporal = pd.merge(_since,_until,how="outer", on=["S", "P", "O"])\
                   .rename(columns={"T_x":"st", "T_y":"ed"})
    # if the occursUntil year does not exist, then fill it with the occursSince year
    _temporal["ed"] = _temporal["ed"].fillna(_temporal["st"])
    
    _temporal["st_year"] = _temporal["st"].str.split("-", expand = True)[0].astype('float')
    _temporal["ed_year"] = _temporal["ed"].str.split("-", expand = True)[0].astype('float')
    
    _temporal = _temporal.assign(st_year_id=np.nan, ed_year_id=np.nan)
    for key, time_idx in sorted(year2id.items(), key=lambda x:x[1]):
        _temporal["st_year_id"] = np.where((_temporal["st_year"] >= key[0]) & 
                                         (_temporal["st_year"] <= key[1]) &
                                         (_temporal["st_year_id"].isna()), 
                                         time_idx, _temporal["st_year_id"])
        _temporal["ed_year_id"] = np.where((_temporal["ed_year"] >= key[0]) & 
                                         (_temporal["ed_year"] <= key[1]) &
                                         (_temporal["ed_year_id"].isna()), 
                                         time_idx, _temporal["ed_year_id"])
    
    # concat non-temporal and temporal data
    _new = pd.concat([_nt, _temporal]).reset_index(drop=True)   
    
    _new["st_year_id"] = _new["st_year_id"].fillna("-999")
    _new["ed_year_id"] = _new["ed_year_id"].fillna("-999")
    _new["st_year_id"] = _new["st_year_id"].astype('int')
    _new["ed_year_id"] = _new["ed_year_id"].astype('int')
    _new["year_interval"] = _new["ed_year_id"] - _new["st_year_id"] + 1
        
    _new["year_interval"] = _new["year_interval"].fillna(1)
    
    _new = _new[_new["year_interval"] > 0]
    
    return _new

train_tmp = preprcessing_m1(train)
valid_tmp = preprcessing_m1(valid)
test_tmp = preprcessing_m1(test)


# generate all year id between interval
def generate_all_timestamp(dataset):
    # repeat records by year_interval
    _tmp = pd.DataFrame(dataset.values.repeat(dataset["year_interval"], axis=0), 
                        columns=dataset.columns)
    # generate all year id
    _tmp["year_id"] = _tmp.groupby(["S","P","O","st_year_id","ed_year_id"])\
                          .cumcount() + _tmp["st_year_id"]    
    _tmp = _tmp[["S","P","O","year_id"]]
    _tmp_key = _tmp[["S","P","O"]].value_counts().reset_index().reset_index()
    _tmp_key = _tmp_key.rename({"index": "key_id", 0: "key_n"}, axis=1)
    _tmp = pd.merge(_tmp, _tmp_key, on=["S","P","O"])
    
    return _tmp

# 124975
train_m1 = generate_all_timestamp(train_tmp)
# 15674
valid_m1 = generate_all_timestamp(valid_tmp)
# 15459
test_m1 = generate_all_timestamp(test_tmp)

# save files
train_m1.to_csv('yago15k-m1/train.txt', sep='\t', header=False, index=False)
valid_m1.to_csv('yago15k-m1/valid.txt', sep='\t', header=False, index=False)
test_m1.to_csv('yago15k-m1/test.txt', sep='\t', header=False, index=False)


### Method 2: merging time modifier
def preprcessing_m2(data):
    dataset = data.copy()
    dataset["timestamp"] = dataset["T"].str.split("-", expand = True)[0].astype('float')
    dataset["timestamp"] = dataset["timestamp"].fillna(-999)
    
    dataset = dataset.assign(year_id=np.nan)
    for key, time_idx in sorted(year2id.items(), key=lambda x:x[1]):
        dataset["year_id"] = np.where((dataset["timestamp"] >= key[0]) & 
                                         (dataset["timestamp"] <= key[1]) &
                                         (dataset["year_id"].isna()), 
                                         time_idx, dataset["year_id"])
    
    dataset["P"] = np.where(dataset["M"] == "<occursSince>", 
                            dataset.agg(lambda x: f"{x['P']}_since", axis=1),
                   np.where(dataset["M"] == "<occursUntil>", 
                            dataset.agg(lambda x: f"{x['P']}_until", axis=1), 
                            dataset["P"]))
    
    dataset["year_id"] = dataset["year_id"].fillna(-999)
    
    return dataset[["S","P","O","year_id"]]

train_m2 = preprcessing_m2(train)
valid_m2 = preprcessing_m2(valid)
test_m2 = preprcessing_m2(test)

# save files
train_m2.to_csv('yago15k-m2/train.txt', sep='\t', header=False, index=False)
valid_m2.to_csv('yago15k-m2/valid.txt', sep='\t', header=False, index=False)
test_m2.to_csv('yago15k-m2/test.txt', sep='\t', header=False, index=False)
