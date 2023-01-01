# %%
#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

# %% [markdown]
# # Introduction
# Though users can automatically run the whole Quant research worklfow based on configurations with Qlib.
# 
# Some advanced users usally would like to carefully customize each component to explore more in Quant.
# 
# If you just want a simple example of Qlib. [Quick start](https://github.com/microsoft/qlib#quick-start) and [workflow_by_code](https://github.com/microsoft/qlib/blob/main/examples/workflow_by_code.ipynb) may be a better choice for you.
# 
# If you want to know more details about Quant research, this notebook may be a better place for you to start.
# 
# We hope this script could be a tutorial for users who are interested in the details of Quant.
# 
# This notebook tries to demonstrate how can we use Qlib to build components step by step. 

# %%
from pprint import pprint
from pathlib import Path
import pandas as pd

# %%
MARKET = "csi300"
BENCHMARK = "SH000300"
EXP_NAME = "tutorial_exp"

# %% [markdown]
# # Data

# %% [markdown]
# ## Get data

# %% [markdown]
# Users can follow [the steps](https://github.com/microsoft/qlib/tree/main/scripts#download-qlib-data) to download data with CLI.
# 
# In this example we use the underlying API to automatically download data

# %%
from qlib.tests.data import GetData
GetData().qlib_data(exists_skip=True)

# %%
import qlib
qlib.init()

# %% [markdown]
# ## Inspect raw data

# %% [markdown]
# Currently, Qlib support several kinds of data source.

# %% [markdown]
# ### Calendar

# %%
from qlib.data import D
D.calendar(start_time='2010-01-01', end_time='2017-12-31', freq='day')[:2]  # calendar data

# %% [markdown]
# ### Basic data

# %%
df = D.features(['SH601216'], ['$open', '$high', '$low', '$close', '$factor'], start_time='2020-05-01', end_time='2020-05-31')   

# %%
import plotly.graph_objects as go
fig = go.Figure(data=[go.Candlestick(x=df.index.get_level_values("datetime"),
                open=df['$open'],
                high=df['$high'],
                low=df['$low'],
                close=df['$close'])])
fig.show()

# %% [markdown]
# ### price adjustment

# %% [markdown]
# Maybe you think the price is not what it looks like in real world.
# 
# Due to the price adjustment, the price will be different from the real trading data .

# %%
import plotly.graph_objects as go
fig = go.Figure(data=[go.Candlestick(x=df.index.get_level_values("datetime"),
                open=df['$open'] / df['$factor'],
                high=df['$high'] / df['$factor'],
                low=df['$low'] / df['$factor'],
                close=df['$close'] / df['$factor'])])
fig.show()

# %% [markdown]
# Please notice the price gap on [2020-05-26](http://vip.stock.finance.sina.com.cn/corp/view/vISSUE_ShareBonusDetail.php?stockid=601216&type=1&end_date=2020-05-20)
# 
# If we want to represent the change of assets value by price, adjust prices are necesary.
# By default, Qlib stores the adjusted prices.

# %% [markdown]
# ### Static universe V.S. dynamic universe

# %% [markdown]
# Dynamic universe

# %%
# dynamic universe
universe = D.list_instruments(D.instruments('csi100'),  start_time='2010-01-01', end_time='2020-12-31')
pprint(universe)

# %%
print(len(universe))

# %% [markdown]
# Qlib use dynamic universe by default.
# 
# csi100 has around 100 stocks each day(it is not that accurate due to the low precision of data).

# %%
df = D.features(D.instruments('csi100'), ['$close'], start_time='2010-01-01', end_time='2020-12-31')   
df.groupby('datetime').size().plot()

# %% [markdown]
# ### Point-In-Time data

# %% [markdown]
# #### download data
# NOTE: To run the test faster, we only download the data of two stocks

# %%
p = Path("~/.qlib/qlib_data/cn_data/financial").expanduser()

# %%
if not p.exists():
    !cd ../../scripts/data_collector/pit/ && pip install -r requirements.txt
    !cd ../../scripts/data_collector/pit/ && python collector.py download_data --source_dir ~/.qlib/stock_data/source/pit --start 2000-01-01 --end 2020-01-01 --interval quarterly --symbol_regex "^(600519|000725).*"
    !cd ../../scripts/data_collector/pit/ && python collector.py normalize_data --interval quarterly --source_dir ~/.qlib/stock_data/source/pit --normalize_dir ~/.qlib/stock_data/source/pit_normalized
    !cd ../../scripts/ && python dump_pit.py dump --csv_path ~/.qlib/stock_data/source/pit_normalized --qlib_dir ~/.qlib/qlib_data/cn_data --interval quarterly
    pass

# %% [markdown]
# #### querying data
# using `roewa(performanceExpressROEWa,业绩快报净资产收益率ROE-加权)` as an example
# 
# If we want to get fundamental data `in the most recent quarter` daily, we can use following example.
# 
# Maitai release part of its fundamental data on [2019-07-13](http://www.cninfo.com.cn/new/disclosure/detail?stockCode=600519&announcementId=1206443183&orgId=gssh0600519&announcementTime=2019-07-13) and  release others on [2019-07-18](http://www.cninfo.com.cn/new/disclosure/detail?stockCode=600519&announcementId=1206456129&orgId=gssh0600519&announcementTime=2019-07-18)

# %%
instruments = ["sh600519"]
data = D.features(instruments, ['P($$roewa_q)'], start_time="2019-01-01", end_time="2019-07-19", freq="day")

# %%
data.tail(15)

# %% [markdown]
# ### experss engine
# 

# %%
D.features(["sh600519"], ['(EMA($close, 12) - EMA($close, 26))/$close - EMA((EMA($close, 12) - EMA($close, 26))/$close, 9)/$close'])

# %% [markdown]
# 
# ## Dataset loading and preprocessing 

# %% [markdown]
# Some heuristic principles of create features
# - make the features comparable between instrumets: remove unit from the features.
# - try to keep the distribution invariant
# - keep the scale of features similar

# %% [markdown]
# ### data loader
# 
# It's interface can be found [here](https://github.com/microsoft/qlib/blob/main/qlib/data/dataset/loader.py#L24) 
# 
# QlibDataLoader is an implementation which load data from Qlib's data source

# %%
from qlib.data.dataset.loader import QlibDataLoader

# %%
qdl = QlibDataLoader(config=(['$close / Ref($close, 10)'], ['RET10']))

# %%
qdl.load(instruments=['sh600519'], start_time='20190101', end_time='20191231')

# %% [markdown]
# ### data handler

# %% [markdown]
# finance data can't be perfect.
# 
# We have to process them before feeding them into Models

# %%
df = qdl.load(instruments=['sh600519'], start_time='20190101', end_time='20191231')

# %%
df.isna().sum()

# %%
df.plot(kind='hist')

# %% [markdown]
# Datahander is responsible for data preprocessing and provides data fetching interface 
# 
# 

# %%
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import ZScoreNorm, Fillna

# %%
# NOTE: normally, the training & validation time range will be  `fit_start_time` ， `fit_end_time`
# however，all the components are decomposed, so the training & validation time range is unknown when preprocessing.
dh = DataHandlerLP(instruments=['sh600519'], start_time='20170101', end_time='20191231',
             infer_processors=[ZScoreNorm(fit_start_time='20170101', fit_end_time='20181231'), Fillna()],
             data_loader=qdl)

# %%
df = dh.fetch()

# %%
df

# %%
df.isna().sum()

# %%
df.plot(kind='hist')

# %% [markdown]
# ### dataset

# %% [markdown]
# #### basic dataset

# %%
from qlib.data.dataset import DatasetH, TSDatasetH

# %%
ds = DatasetH(dh, segments={"train": ('20180101', '20181231'), "valid": ('20190101', '20191231')})

# %%
ds.prepare('train')

# %%
ds.prepare('valid')

# %% [markdown]
# #### Time Series Dataset
# 
# For different model, the required dataset format will be different.
# 
# For example, Qlib provides a Time Series Dataset(TSDatasetH) to help users to create time-series dataset.

# %%
ds = TSDatasetH(step_len=10, handler=dh, segments={"train": ('20180101', '20181231'), "valid": ('20190101', '20191231')})
train_sampler = ds.prepare('train')

# %%
train_sampler

# %%
train_sampler[0] # Retrieving the first example

# %%
train_sampler['2018-01-08', 'sh600519']  # get the time series by <'timestamp', 'instrument_id'> index

# %% [markdown]
# ### Off-the-shelf dataset
# 
# Qlib integrated some dataset alreadly

# %%
handler_kwargs = {
        "start_time": "2008-01-01",
        "end_time": "2020-08-01",
        "fit_start_time": "2008-01-01",
        "fit_end_time": "2014-12-31",
        "instruments": MARKET,
}
handler_conf = {
    "class": "Alpha158",
    "module_path": "qlib.contrib.data.handler",
    "kwargs": handler_kwargs,
}
pprint(handler_conf)

# %%
from qlib.utils import init_instance_by_config

# %%
hd = init_instance_by_config(handler_conf)

# %% [markdown]
# Using config to create instance is a highly frequently used practice in Qlib (e.g. the [workflows configurations](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml) are based on it).
# 
# 
# The above configuration is the same as the code below

# %%
from qlib.contrib.data.handler import Alpha158
hd = Alpha158(**handler_kwargs)

# %% [markdown]
# This dataset has the same structure as the simple one with 1 column  we created just now.

# %%
df = hd.fetch()

# %%
df

# %%
hd.data_loader

# %%
hd.data_loader.fields

# %% [markdown]
# #### some details
# 
# The training data may not be the same as the test data.
# 
# e.g.
# - the training dataset and test dataset use a different fitlering rules,  data processing

# %%
hd.learn_processors

# %%
hd.infer_processors

# %%
hd.process_type # appending type

# %%
hd.fetch(col_set="label", data_key=hd.DK_L)

# %%
hd.fetch(col_set="label", data_key=hd.DK_I)

# %%
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

# %%
dataset = init_instance_by_config(dataset_conf)

# %% [markdown]
# # Model Training & Inference
# 
# [Model interface](https://github.com/microsoft/qlib/blob/main/qlib/model/base.py)

# %%
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord

# %%
model = init_instance_by_config({
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
        },
})

# %%
# start exp to train model
with R.start(experiment_name=EXP_NAME):
    model.fit(dataset)
    R.save_objects(trained_model=model)

    rec = R.get_recorder()
    rid = rec.id # save the record id

    # Inference and saving signal
    sr = SignalRecord(model, dataset, rec)
    sr.generate()

# %% [markdown]
# # Evaluation:
# - Signal-based
# - Portfolio-based: backtest 

# %%
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

# backtest and analysis
with R.start(experiment_name=EXP_NAME, recorder_id=rid, resume=True):

    # signal-based analysis
    rec = R.get_recorder()
    sar = SigAnaRecord(rec)
    sar.generate()
    
    #  portfolio-based analysis: backtest
    par = PortAnaRecord(rec, port_analysis_config, "day")
    par.generate()

# %% [markdown]
# # Loading results & Analysis

# %% [markdown]
# ## loading data
# Because Qlib leverage MLflow to save model & data.
# All the data can be access by `mlflow ui`

# %%
# load recorder
recorder = R.get_recorder(recorder_id=rid, experiment_name=EXP_NAME)

# %%
# load previous results
pred_df = recorder.load_object("pred.pkl")
report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

# %%
# Previous Model can be loaded. but it is not used.
loaded_model = recorder.load_object("trained_model")
loaded_model

# %%
from qlib.contrib.report import analysis_model, analysis_position

# %% [markdown]
# ## analysis position

# %% [markdown]
# ### report

# %%
analysis_position.report_graph(report_normal_df)

# %% [markdown]
# ### risk analysis

# %%
analysis_position.risk_analysis_graph(analysis_df, report_normal_df)

# %% [markdown]
# ## analysis model

# %%
label_df = dataset.prepare("test", col_set="label")
label_df.columns = ['label']

# %% [markdown]
# ### score IC

# %%
pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
analysis_position.score_ic_graph(pred_label)

# %% [markdown]
# ### model performance

# %%
analysis_model.model_performance_graph(pred_label)



# %%
