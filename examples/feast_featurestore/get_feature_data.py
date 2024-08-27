#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 23:05:11 2024

@author: edward
"""
import pandas as pd
from feast import FeatureStore
import datetime

store = FeatureStore("./")

# Define the entity dataframe
query_user_id = 412120092
query_timestamp = datetime.datetime.fromisoformat("2019-12-04 20:18:05+00:00")
entity_df = pd.DataFrame({
    "user_id": [query_user_id], # join key
    # looking back from current timestamp, must be named as event_timestamp
    "event_timestamp": [query_timestamp],
})
feature_list = [
    "user_sessions:product_id",
    "user_sessions:event_type",
    "user_sessions:event_time"
]

# %%Retrieve offline historical features
# nearest, newest event that is older than the current event_timestamp
# that is, the freshest feature
features = store.get_historical_features(
    entity_df=entity_df,
    features=feature_list,
    full_feature_names=False
).to_df()

# %% Retrieve Online featurestore
features = store.get_online_features(
    entity_rows=[{"user_id": query_user_id}],
    features=feature_list
).to_dict()