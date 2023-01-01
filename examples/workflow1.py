#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   workflow1.py
@Time    :   2022/12/30 12:21:57
@Author  :   Tianlong Wang
@Contact :   tianlongwang13@gmail.com
@License :   (C)Copyright 2000-2022, Tianlong Wang
'''
#%%
import pandas as pd
import qlib

qlib.init()

from qlib.data import D

from qlib.utils import init_instance_by_config
#%%

MARKET = "csi300"
BENCHMARK = "SH000300"
EXP_NAME = "TLWDumpExpLinear"

handler_kwargs = {
        "start_time": "2018-01-01",
        "end_time": "2020-08-01",
        "fit_start_time": "2018-01-01",
        "fit_end_time": "2019-12-31",
        "instruments": MARKET,
}
handler_conf = {
    "class": "Alpha158",
    "module_path": "qlib.contrib.data.handler",
    "kwargs": handler_kwargs,
}



#%%
hd = init_instance_by_config(handler_conf)

#%%



model_config = { "class": "LinearModel",
            "module_path": "qlib.contrib.model.linear",
            "kwargs": {
                "estimator": "ols",
            },
        }

# model = init_instance_by_config({
#         "class": "LGBModel",
#         "module_path": "qlib.contrib.model.gbdt",
#         "kwargs": {
#             "loss": "mse",
#             "colsample_bytree": 0.8879,
#             "learning_rate": 0.0421,
#             "subsample": 0.8789,
#             "lambda_l1": 205.6999,
#             "lambda_l2": 580.9768,
#             "max_depth": 3,
#             "num_leaves": 20,
#             "num_threads": 4,
#         },
# })

#%%
dataset_conf = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": hd,
            "segments": {
                "train": ("2008-01-01", "2014-12-31"),
                "valid": ("2015-01-01", "2016-12-31"),
                "test": ("2017-01-01", "2020-08-01"),
            },
        },
}
dataset = init_instance_by_config(dataset_conf)
#%%

from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord

# start exp to train model
with R.start(experiment_name=EXP_NAME):
    model.fit(dataset)
    R.save_objects(trained_model=model)

    rec = R.get_recorder()
    rid = rec.id # save the record id

    # Inference and saving signal
    sr = SignalRecord(model, dataset, rec)
    sr.generate()
#%%

###################################
# prediction, backtest & analysis
###################################
port_analysis_config = {
    "executor": {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    },
    "strategy": {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "kwargs": {
            "signal": "<PRED>",
            "topk": 50,
            "n_drop": 5,
        },
    },
    "backtest": {
        "start_time": "2017-01-01",
        "end_time": "2020-08-01",
        "account": 100000000,
        "benchmark": BENCHMARK,
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    },
}





#%%
# backtest and analysis
with R.start(experiment_name=EXP_NAME, recorder_id=rid, resume=True):

    # signal-based analysis
    rec = R.get_recorder()
    sar = SigAnaRecord(rec)
    sar.generate()
    
    #  portfolio-based analysis: backtest
    par = PortAnaRecord(rec, port_analysis_config, "day")
    par.generate()

#%%

recorder = R.get_recorder(recorder_id=rid, experiment_name=EXP_NAME)

# load previous results
pred_df = recorder.load_object("pred.pkl")
report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

#%%
# Previous Model can be loaded. but it is not used.
loaded_model = recorder.load_object("trained_model")
#%%
from qlib.contrib.report import analysis_model, analysis_position

analysis_position.report_graph(report_normal_df)
#%%

label_df = dataset.prepare("test", col_set="label")
label_df.columns = ['label']

#%%
pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
#%%
analysis_position.score_ic_graph(pred_label)
#%%

analysis_model.model_performance_graph(pred_label)