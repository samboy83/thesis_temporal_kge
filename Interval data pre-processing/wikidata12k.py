# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import os, pdb
from collections import defaultdict as dd


os.chdir("D:/MU/05.HWS2021/Thesis/tmp-kge/data")

# data loading
# 32497
train = pd.read_table('wikidata12k/train.txt', sep = '\t', header = None,
                        names = ['S','P','O','st','ed'])
# 4062
valid = pd.read_table('wikidata12k/valid.txt', sep = '\t', header = None,
                        names = ['S','P','O','st','ed'])
# 4062
test = pd.read_table('wikidata12k/test.txt', sep = '\t', header = None,
                        names = ['S','P','O','st','ed'])


YEARMAX = 2020
YEARMIN = 0

# create year2id 
# only year is considered
def create_year2id(triple_time, bin_size=300):
    year2id = dict()
    freq = dd(int)
    count = 0
    year_list = []

    for k, v in triple_time.items():
        try:
            start = v[0].split('-')[0]
            end = v[1].split('-')[0]
        except:
            pdb.set_trace()

        if start.find('#') == -1 and len(start) == 4: year_list.append(int(start))
        if end.find('#') == -1 and len(end) == 4: year_list.append(int(end))

    year_list.sort()
    for year in year_list:
        freq[year] = freq[year] + 1

    year_class = []
    count = 0
    for key in sorted(freq.keys()):
        count += freq[key]
        if count > bin_size:
            year_class.append(key)
            count = 0
    prev_year = 0
    i = 0
    for i, yr in enumerate(year_class):
        year2id[(prev_year, yr)] = i
        prev_year = yr + 1
    year2id[(prev_year, max(year_list))] = i + 1    

    return year2id


triple_time = dict()
count = 0
for line in train.itertuples():    
    triple_time[count] = [x.split('-')[0] for x in [line.st, line.ed]]
    count += 1

year2id = create_year2id(triple_time, bin_size=300)  # (bin_start, bin_end) to id


### Method 1: All year information generaion 
def preprcessing_m1(data, year2id):
    dataset = data.copy()
    # if start date does not exist, set it to YEARMIN
    dataset.loc[dataset['st'] == "####-##-##", "st"] = str(YEARMIN) + "-##-##"
    # if end date does not exist, set it to YEARMAX    
    dataset.loc[dataset['ed'] == "####-##-##", "ed"] = str(YEARMAX) + "-##-##"
    
    # create year
    dataset["st_year"] = dataset["st"].str[:-6].astype('int')
    dataset["ed_year"] = dataset["ed"].str[:-6].astype('int')
    
    
    # if the start year is later than the end year, then delete     
    dataset = dataset[dataset["st_year"] <= dataset["ed_year"]]
    
    # convert year to year id
    dataset = dataset.assign(st_year_id=np.nan, ed_year_id=np.nan)
    for key, time_idx in sorted(year2id.items(), key=lambda x:x[1]):
        dataset["st_year_id"] = np.where((dataset["st_year"] >= key[0]) & 
                                         (dataset["st_year"] <= key[1]) &
                                         (dataset["st_year_id"].isna()), 
                                         time_idx, dataset["st_year_id"])
        dataset["ed_year_id"] = np.where((dataset["ed_year"] >= key[0]) & 
                                         (dataset["ed_year"] <= key[1]) &
                                         (dataset["ed_year_id"].isna()), 
                                         time_idx, dataset["ed_year_id"])
    
    dataset["st_year_id"] = dataset["st_year_id"].astype('int')
    dataset["ed_year_id"] = dataset["ed_year_id"].astype('int')
    dataset["interval"] = dataset["ed_year_id"] - dataset["st_year_id"] + 1
    
    return dataset

train_tmp = preprcessing_m1(train, year2id)
valid_tmp = preprcessing_m1(valid, year2id)
test_tmp = preprcessing_m1(test, year2id)


# generate all year id between interval
def generate_all_timestamp(dataset):
    # repeat records by year_interval
    _tmp = pd.DataFrame(dataset.values.repeat(dataset["interval"], axis=0), 
                        columns=dataset.columns)
    # generate all year id
    _tmp["year_id"] = _tmp.groupby(["S","P","O","st_year","ed_year"])\
                          .cumcount() + _tmp["st_year_id"]    
    _tmp = _tmp[["S","P","O","year_id"]]
    _tmp_key = _tmp[["S","P","O"]].value_counts().reset_index().reset_index()
    _tmp_key = _tmp_key.rename({"index": "key_id", 0: "key_n"}, axis=1)
    _tmp = pd.merge(_tmp, _tmp_key, on=["S","P","O"])
    
    return _tmp

# 272364
train_m1 = generate_all_timestamp(train_tmp)
# 22696
valid_m1 = generate_all_timestamp(valid_tmp)
# 21551
test_m1 = generate_all_timestamp(test_tmp)

# save files
train_m1.to_csv('wikidata12k-m1/train.txt', sep='\t', header=False, index=False)
valid_m1.to_csv('wikidata12k-m1/valid.txt', sep='\t', header=False, index=False)
test_m1.to_csv('wikidata12k-m1/test.txt', sep='\t', header=False, index=False)


### Method 2: merging time modifier
def preprcessing_m2(data):
    dataset = data.copy()
    # if start date does not exist, set it to YEARMIN
    dataset.loc[dataset['st'] == "####-##-##", "st"] = str(YEARMIN) + "-##-##"
    # if end date does not exist, set it to YEARMAX    
    dataset.loc[dataset['ed'] == "####-##-##", "ed"] = str(YEARMAX) + "-##-##"
    
    # create year
    dataset["st_year"] = dataset["st"].str[:-6].astype('int')
    dataset["ed_year"] = dataset["ed"].str[:-6].astype('int')
    
    
    # if the start year is later than the end year, then delete     
    dataset = dataset[dataset["st_year"] <= dataset["ed_year"]]
    
    # convert year to year id
    dataset = dataset.assign(st_year_id=np.nan, ed_year_id=np.nan)
    for key, time_idx in sorted(year2id.items(), key=lambda x:x[1]):
        dataset["st_year_id"] = np.where((dataset["st_year"] >= key[0]) & 
                                         (dataset["st_year"] <= key[1]) &
                                         (dataset["st_year_id"].isna()), 
                                         time_idx, dataset["st_year_id"])
        dataset["ed_year_id"] = np.where((dataset["ed_year"] >= key[0]) & 
                                         (dataset["ed_year"] <= key[1]) &
                                         (dataset["ed_year_id"].isna()), 
                                         time_idx, dataset["ed_year_id"])
    
    dataset["st_year_id"] = dataset["st_year_id"].astype('int')
    dataset["ed_year_id"] = dataset["ed_year_id"].astype('int')
        
    # repeat records 
    _tmp = pd.DataFrame(dataset.values.repeat(2, axis=0), 
                        columns=dataset.columns)
    _tmp["cnt"] = _tmp.groupby(["S","P","O","st_year","ed_year"]).cumcount()
    
    # merge time modifier into predicate
    _tmp["P"] = np.where(_tmp["cnt"] == 1, 
                         _tmp.agg(lambda x: f"{x['P']}_since", axis=1), 
                         _tmp.agg(lambda x: f"{x['P']}_until", axis=1))
    _tmp["timestamp"] = np.where(_tmp["cnt"] == 1, _tmp["st_year_id"], _tmp["ed_year_id"])
    
    return _tmp[["S", "P", "O", "timestamp"]]

train_m2 = preprcessing_m2(train)
valid_m2 = preprcessing_m2(valid)
test_m2 = preprcessing_m2(test)


# save files
train_m2.to_csv('wikidata12k-m2/train.txt', sep='\t', header=False, index=False)
valid_m2.to_csv('wikidata12k-m2/valid.txt', sep='\t', header=False, index=False)
test_m2.to_csv('wikidata12k-m2/test.txt', sep='\t', header=False, index=False)
