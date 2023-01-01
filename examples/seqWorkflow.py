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

if __name__ == "__main__":
    hd = init_instance_by_config(handler_conf)

    model = init_instance_by_config({
            "class": "Linear",
            "module_path": "qlib.contrib.model.linear",
            "kwargs": {
                "estimator": "ols",
            },
    })