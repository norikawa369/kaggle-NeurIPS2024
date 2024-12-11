import pandas as pd
import numpy as np
import pickle
import os
import warnings

warnings.filterwarnings('ignore')

with open(os.path.join('/work/ge58/e58004/kaggle-comps/kaggle-leashbio-belka/input/custom_data', 'bblocks.pkl'), mode='rb') as f:
    train_bbs = pickle.load(f)

test = pd.read_csv('/work/ge58/e58004/kaggle-comps/kaggle-leashbio-belka/input/test.csv')

known = 0
for k in [1, 2, 3]:
    known += test['buildingblock%d_smiles' % k].apply(lambda x: x in train_bbs).astype(int)

test['known'] = known

# In test set, 3 building blocks are either all known or all unknown
print('known', known.unique())

# 66% have known building blocks
print('known fraction:', (known == 3).mean())
print('unknown       :', (known == 0).mean())

# subver1 = pd.read_csv('/work/ge58/e58004/kaggle-comps/kaggle-leashbio-belka/exp/exp1/output/ver1_wist_saved_graph/submission.csv')
# subver2_part = pd.read_csv('/work/ge58/e58004/kaggle-comps/kaggle-leashbio-belka/exp/exp1/output/ver2/submission.csv')
subver1 = pd.read_csv('/work/ge58/e58004/kaggle-comps/kaggle-leashbio-belka/exp/exp3/output/base/submission.csv')
subver2 = pd.read_csv('/work/ge58/e58004/kaggle-comps/kaggle-leashbio-belka/exp/exp3/output/lr1e-4_graphdim128/submission.csv')
subver3 = pd.read_csv('/work/ge58/e58004/kaggle-comps/kaggle-leashbio-belka/exp/exp3/output/lr1e-4_layer5/submission.csv')

test = test.merge(subver1[['id', 'binds']], how='left', on='id')
test = test.rename(columns={'binds': 'binds_ver1'})

test = test.merge(subver2[['id', 'binds']], how='left', on='id')
test = test.rename(columns={'binds': 'binds_ver2'})

test = test.merge(subver3[['id', 'binds']], how='left', on='id')
test = test.rename(columns={'binds': 'binds_ver3'})

# test_share = test.loc[test['known']==3]
# test_nonshare = test.loc[test['known']==0]

test['binds'] = (test['binds_ver1'] + test['binds_ver2'] +test['binds_ver3']) / 3
# test_share['binds'] = (test_share['binds_ver1'] + test_share['binds_ver2']) / 2
# test_nonshare['binds'] = test_nonshare['binds_ver2']

# sub = pd.concat([test_share[['id', 'binds']], test_nonshare[['id', 'binds']]], ignore_index=True)
# print(f'sub_df has {sub.isnull().sum()} nulls.')
# print(f'sample_sub has {len(sub)} rows.')

sub = test[['id', 'binds']]
print(f'sub_df has {sub.isnull().sum()} nulls.')
print(f'sample_sub has {len(sub)} rows.')



print('save the submission.csv')
save_dir = os.path.join('/work/ge58/e58004/kaggle-comps/kaggle-leashbio-belka/exp/exp3/output/ensemble', 'exp3_all')
os.makedirs(save_dir, exist_ok=True)
sub.to_csv(os.path.join(save_dir, 'submission.csv'), index=False)