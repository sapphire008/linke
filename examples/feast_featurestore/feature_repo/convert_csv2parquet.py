#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 19:55:21 2024

@author: edward
"""
import os
import glob
import dask.dataframe as dd

base_dir = "/Users/edward/Documents/Scripts/kubeflow-components/examples/feast_featurestore/feature_repo/data/eCommerce-behavior-data-from-multi-category-store"
csv_files = glob.glob(os.path.join(base_dir, "*.csv"))


# %% Convert to parquet
df = dd.read_csv(os.path.join(base_dir, "*.csv"), parse_dates=[0], dtype={'category_code': 'object'})
df.to_parquet(os.path.join(base_dir, "dataset"))


#%% Check
import pyarrow.dataset as ds
dataset = ds.dataset(os.path.join(base_dir, "dataset"), format="parquet")