import re
import time
import citybrain_platform
from odps import ODPS 
from tqdm import tqdm
import time
import numpy as np
import argparse
import json
import time
from odps.models import Instance
from odps.models import TableSchema, Column, Partition
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import platform
import psutil
import pandas as pd
import xarray as xr
import cftime 
import gc
import os
import sys
import warnings
warnings.filterwarnings("ignore")


#------------------------------（开源代码版本）本地使用pandas实现热浪-----------------------------------

def load_df(file_path):
    """ 
    输入:csv文件路径
    输出:pandas dataframe
    """

    print("Start to load csv", file_path)
    time_point = time.time()
    df_raw = pd.read_csv(file_path)
    print("It takes", time.time()-time_point, "to load csv")
    return df_raw

# function for getting mean value of each grid for each day 
def get_mean(df_raw, model_type):
    """ 
    输入:load_df函数的输出,pandas dataframe
    输出:将每个格点每天的值取平均后的pandas dataframe以及添加了datetime列的pandas dataframe
    """
    if isinstance(df_raw_mean["time"][0],cftime._cftime.DatetimeNoLeap):
        df_raw_mean["time"] = df_raw_mean["time"].apply(lambda x: str(x))
    df_raw_mean["datetime"]=(pd.to_datetime(df_raw_mean["time"],errors="coerce"))
    # calculate mean
    if model_type == "cesm":
        df_raw_mean = df_raw.set_index(["lat","lon","time"]).mean(axis=1).reset_index().rename(columns={0:"mean"})
        # convert date to datetime
        # if isinstance(df_raw_mean["time"][0],cftime._cftime.DatetimeNoLeap):
        #     df_raw_mean["time"] = df_raw_mean["time"].apply(lambda x: str(x))
        # df_raw_mean["datetime"]=(pd.to_datetime(df_raw_mean["time"],errors="coerce"))
    else:
        # For the latest version, I used "time" rather than date for CMIP
        #df_raw_mean = df_raw.set_index(["lat","lon","date"]).mean(axis=1).reset_index().rename(columns={0:"mean"})
        # convert date to datetime
        #df_raw_mean["datetime"]=(pd.to_datetime(df_raw_mean["date"]))
        
        df_raw_mean = df_raw.set_index(["lat","lon","time"]).mean(axis=1).reset_index().rename(columns={0:"mean"})
        # convert date to datetime
        # if isinstance(df_raw_mean["time"][0],cftime._cftime.DatetimeNoLeap):
        #     df_raw_mean["time"] = df_raw_mean["time"].apply(lambda x: str(x))
        # df_raw_mean["datetime"]=(pd.to_datetime(df_raw_mean["time"],errors="coerce"))
        
    # sort datetime
    df_raw_date_sort = df_raw_mean[["lat","lon","datetime","mean"]].sort_values(["lat","lon","datetime"]).reset_index(drop=True)
    return df_raw_date_sort

# function for calculating single model quantile value
def single_model_quant(df, set_quantile, variable):
    """
    输入: get_mean函数的输出,pandas dataframe
    输出: 每个格点的quantile值,pandas dataframe
    """
    quant_value = df.groupby(['lat','lon'])[variable].quantile(q = set_quantile,interpolation='linear').reset_index().rename(columns={variable: "quant"})[["lat","lon","quant"]]
    return quant_value

"""
# function for calculating multi-model quantile value
def multi_model_quant(df, set_quantile, model_type):
    if model_type == "cesm":
        df_drop_time = df.drop(["time"], axis = 1)
        quant_value = df_drop_time.groupby(['lat','lon']).quantile([set_quantile]).reset_index().drop(["level_2"],axis=1)
    else:
        df_drop_time = df.drop(["date"], axis = 1)
        quant_value = df_drop_time.groupby(['lat','lon']).quantile([set_quantile]).reset_index().drop(["level_2"],axis=1)
    return quant_value

# Get and save quantile
def save_quantile(df, set_quantile, variable, model_type, save_csv):
    if variable != None:
        quant_value = single_model_quant(df, set_quantile, variable)
    else:
        quant_value = multi_model_quant(df, set_quantile, model_type)

    if save_csv != None:
        quant_value.to_csv(save_csv, index=False)

    return quant_value
"""

# useful function for counting duration for each heat wave groups
def get_cont_groups(df):
    """
    输入: get_mean函数的输出,pandas dataframe
    输出: 带有group_id的pandas dataframe,其中首先筛选出HW=0的行,然后将连续的HW=0的行分为一组,进行cumsum操作,得到group_id
    """
    df_HW = df[df["HW"]==0].reset_index(drop=True)
    dt = df_HW['datetime']
    day = pd.Timedelta('1d')
    in_block = ((dt - dt.shift(-1)).abs() == day) | (dt.diff() == day)
    breaks = dt.diff() != day
    df_HW["group_id"] = breaks.cumsum()
    return df_HW

# get heat waves dataframe for 2006
def get_heat_waves_df(df, set_quantile, duration_threshold, model_type, quantile_avail):
    """
    输入: df_raw_date_sort, pandas dataframe
    输出:
        1. heat wave dataframe, pandas dataframe,其中包含了每个格点中符合热浪定义的事件以及其对应的group_id,以及每个group_id的duration
        2. quantile dataframe, pandas dataframe,其中包含了每个格点的quantile值
    """
    print("The quantile is:", set_quantile)
    print("The duration threshold is:", duration_threshold)
    # get mean value of each grid for each day
    mean_value = get_mean(df, model_type)
    
    if set_quantile != None:
        # get quantile value of each grid from the mean value time series
        quant_value = single_model_quant(mean_value, set_quantile, "mean")
    else:
        quant_value = quantile_avail

    # initialize no heatwaves as "1"
    mean_value["HW"] = 1
    # merge df with quantile
    df_with_quantile = pd.merge(mean_value, quant_value, how="left", on=['lat', 'lon'])
    # get heat wave signal, "0" means temperature > threshold
    df_with_quantile["HW"][df_with_quantile["mean"]>= df_with_quantile["quant"]] = 0
    # get groups of heatwaves
    df_HW = get_cont_groups(df_with_quantile)
    # calculate duration for each heat wave groups
    df_count = pd.DataFrame(df_HW["group_id"].value_counts()).reset_index().rename(columns={"group_id": "duration","index": "group_id"})
 #   df_count = pd.DataFrame(df_HW["group_id"].value_counts()).reset_index().rename(columns={"count": "duration","index": "group_id"})
    # get groups and their duration
    df_HW_count = pd.merge(df_HW,df_count,how="left",on=['group_id'])
    # select "duration > duration_threshold" as our final HW dataframe
    df_HW_final = df_HW_count[df_HW_count["duration"]>=duration_threshold].set_index(["lat","lon"])

    return df_HW_final, quant_value

# get frequency from the output of the "get_heat_waves_df"
def get_frequency(df_HW_final,model_name):
    """
    输入: get_heat_waves_df函数的输出,pandas dataframe
    输出: 每个格点的frequency值,pandas dataframe
    """
    df_HW_final_group = df_HW_final["group_id"].drop_duplicates().reset_index()
    df_HW_final_group["num"] = 1
    df_HW_freq=df_HW_final_group.groupby(["lat","lon"]).sum()["num"]
    df_HW_freq = pd.DataFrame(df_HW_freq).reset_index()
    df_HW_freq[model_name] =  df_HW_freq["num"]/10
    df_HW_freq_final = df_HW_freq[["lat","lon",model_name]].set_index(["lat","lon"])

    return df_HW_freq_final

# get duration from the output of the "get_heat_waves_df"
def get_duration(df_HW_final,model_name):
    """
    输入: get_heat_waves_df函数的输出,pandas dataframe
    输出: 每个格点的duration值(每个地区单一事件持续时间之和),pandas dataframe
    """
    df_HW_duration = df_HW_final[["group_id","duration"]].drop_duplicates()["duration"].reset_index().groupby(["lat","lon"]).sum()["duration"]
    df_HW_duration = pd.DataFrame(df_HW_duration).reset_index()
    df_HW_duration[model_name] = df_HW_duration["duration"]/10
    df_HW_duration_final = df_HW_duration[["lat","lon",model_name]].set_index(["lat","lon"])

    return df_HW_duration_final

# get intensity from the output of the "get_heat_waves_df"
def get_intensity(df_HW_final,model_name):
    """
    输入: get_heat_waves_df函数的输出,pandas dataframe
    输出: 每个格点的intensity值(每个地区单一事件平均强度之和),pandas dataframe
    """
    df_HW_intensity = pd.DataFrame(df_HW_final.reset_index().groupby(["lat","lon"]).mean()["mean"]).reset_index()
    df_HW_intensity[model_name] = df_HW_intensity["mean"]
    df_HW_intensity_final = df_HW_intensity[["lat","lon",model_name]].set_index(["lat","lon"])

    return df_HW_intensity_final




#------------------------------pandas basic function-----------------------------------

def show_now_time():
    """
    作用：显示当前时间
    """
    print('now time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))


def show_computer_info():
    """
    作用：显示电脑的基本信息
    """
    time1 = time.time()
    print('time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time1)))
    print('-'*50)
    # system
    print('\nOS')
    print('Operating System Name:', platform.system())
    print('Operating System Version:', platform.release())
    print('Bit:', platform.machine())
    print('-'*50)
    # CPU
    print('\nCPU:')
    print('CPU model:', platform.processor())
    print('Physical cores:', psutil.cpu_count(logical=False))
    print('Total cores:', psutil.cpu_count(logical=True))
    print('-'*50)

    # 获取内存信息
    print('\nMemory')
    mem = psutil.virtual_memory()
    print('Total memory:', round(mem.total / 1024**3), 'GB')
    print('Available memory:', round(mem.available / 1024**3), 'GB')
    print('-'*50)

    # disk
    print('\nDisk')
    disk = psutil.disk_usage('/')
    print('Total disk space:', round(disk.total / (1024**3)), 'GB')
    print('Used disk space:', round(disk.used / (1024**3)), 'GB')

def show_process_info():
    """
    作用：显示当前进程的基本信息
    """
    # 记录开始时间
    start_time = time.time()
    print('-'*50)
    print('start time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    # 记录开始时的CPU和内存使用情况  
    start_cpu_usage = psutil.cpu_percent()  
    start_memory_usage = psutil.virtual_memory().used  
    print('start cpu usage: ', start_cpu_usage, '%')
    print('start memory usage: ', start_memory_usage/(1024*1024*1024), 'GB')

    # 记录开始时的进程数
    start_process_num = len(psutil.pids())
    print('start process num: ', start_process_num)
    return start_time,start_cpu_usage,start_memory_usage,start_process_num

def get_process_diff(start_time,start_cpu_usage,start_memory_usage,start_process_num,
                     end_time = None,end_cpu_usage = None,end_memory_usage = None,end_process_num = None):
    """
    作用：显示当前进程的基本信息以及与开始时的差值
    """
    if end_time is None:
        end_time = time.time()
    if end_cpu_usage is None:
        end_cpu_usage = psutil.cpu_percent()
    if end_memory_usage is None:
        end_memory_usage = psutil.virtual_memory().used
    if end_process_num is None:
        end_process_num = len(psutil.pids())

    print('end time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    print('end cpu usage: ', end_cpu_usage, '%')
    print('end memory usage: ', end_memory_usage/(1024*1024*1024), 'GB')
    print('end process num: ', end_process_num)
    # print diff
    print('time diff: ', (end_time - start_time), ' seconds')
    print('cpu diff: ', (end_cpu_usage - start_cpu_usage), '%')
    print('memory diff: ', (end_memory_usage - start_memory_usage)/(1024*1024*1024), 'GB')
    print('process diff: ', end_process_num - start_process_num, ' processes')




def load_zarr_data(type_id = 6,start_time = 0,end_time = 3650,show_all = False):
    """
        type_id: 0-10,对应不同的原始数据集
        start_time: 0
        end_time: 0-3650default: 3650,也即10年,2006-2015,最大为36500,也即100年
        show_all: True/False,是否显示所有的数据集,默认为False,若为True,则不会加载数据,而是显示所有的数据集
    """
    type_list = ['FLNS.zarr', 'FSNS.zarr', 'PRECSC.zarr', 'PRECSL.zarr', 'PRECT.zarr', 'QBOT.zarr',
                     'TREFHTMX.zarr', 'TREFHT.zarr', 'TREFMXAV_U.zarr', 'UBOT.zarr', 'VBOT.zarr']
    if show_all:
        print(type_list)
        return None
    # show time

    print('type: ', type_list[type_id])
    data_type = type_list[type_id]
    start_t = time.time()
    print('start time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_t)))
    data_path = '/datadisk/cesm_raw/'+data_type
    ds = xr.open_zarr(data_path).isel(time=slice(start_time,end_time))
    # ds = ds.assign_coords(time = ds.indexes["time"].to_datetimeindex())
    end_t = time.time()
    load_time = end_t- start_t
    print('end time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_t)))
    print('load_time: ',  round(load_time,2), ' seconds to finish.')
    return ds,load_time

def transform2df(ds,member_id = 2):
    """
        ds: xarray dataset
        member_id: [1-35,101-105],对应不同的member_id,默认为2
    """
    # 如果member_id为list,默认格式为[1,35,1]，也即从1到35，步长为1
    if isinstance(member_id, list):
        # 如果member_id为1D list,统一转化为2D list
        if not isinstance(member_id[0], list):
            member_id_raw = [member_id]
        # 如果member_id为2D list,则直接赋值
        else:
            member_id_raw = member_id
        # 将member_id_raw（2D list）转化为member_id (1D list)
        member_id = []
        for i in range(len(member_id_raw)):
            member_id += list(np.arange(member_id_raw[i][0],member_id_raw[i][1],member_id_raw[i][2]))
    start_time = time.time()
    print('-'*50)
    print('transform start time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    ds1 = ds['TREFHTMX'].sel(member_id=member_id)
    df = ds1.to_dataframe().reset_index().drop(columns=['member_id'])
    #df = df.sort_values(by = 'time')
    df = df.sort_values(by = ['lon','lat','time'])
    end_time =time.time()
    transform_time = time.time() - start_time
    print('transform end time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    print('it took ', round(transform_time ,2), ' seconds to finish transform.')
    return df,transform_time

def cal_heat_waves(df,set_quantile = 0.98,duration_threshold = 3,quantile_avail = None,model_type= 'some_model'):
    """
    输入：
        df: pandas dataframe
        set_quantile: 设置quantile值,默认为0.98
        duration_threshold: 设置duration阈值,默认为3
        quantile_avail:默认为None, 如果quantile_avail不为None,则不会计算set_quantile,而是直接使用quantile_avail(绝对温度值)
        model_type: 'cesm'/'cmip',默认为'cesm'
    输出：
        1. heat wave dataframe, pandas dataframe,其中包含了每个格点中符合热浪定义的事件以及其对应的group_id,以及每个group_id的duration
        2. quantile dataframe, pandas dataframe,其中包含了每个格点的quantile值
        3. take_time0: 计算heat wave dataframe的时间
    """

    start_time = time.time()
    print('-'*50)
    print('cal_heat_waves start time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    res,quant_value = get_heat_waves_df(df = df,set_quantile = set_quantile,duration_threshold = duration_threshold,
                                                    quantile_avail = quantile_avail,model_type= model_type)

    end_time =time.time()
    take_time0 = time.time() - start_time
    print('cal_heat_waves end time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    print('it took ', round(take_time0,2), ' seconds to finish cal heat waves.')
    return res,quant_value,take_time0

def cal_metrics(df,var_name,is_merge = False,model_name = 'some_model'):
    """
    输入：
        df: pandas dataframe
        var_name: 变量名,默认为'TREFHTMX'
        is_merge: 是否合并,默认为False
        model_name: 模型名,默认为'some_model'
    输出：
        如果is_merge为True,则输出合并所有指标的df_final,以及frequency,duration,intensity的pandas dataframe,总时间,以及计算frequency,duration,intensity的时间
        如果is_merge为False,则只输出指标 frequency,duration,intensity的pandas dataframe,总时间,以及计算frequency,duration,intensity的时间
    """
    print('-'*50)
    start_time = time.time()
    print('-'*50)
    print('cal_metrics start time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    frequency = get_frequency(df_HW_final= df,model_name=model_name)
    end_time =time.time()
    take_time1 = time.time() - start_time
    print('cal_fre end time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    print('it took ', round(take_time1,2), ' seconds to calculate frequency.')
    print()

    start_time = time.time()
    print('start time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    duration = get_duration(df_HW_final= df,model_name=model_name)
    end_time =time.time()
    take_time2 = time.time() - start_time
    print('cal_duration end time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    print('it took ', round(take_time2,2), ' seconds to calculate duration.')
    print()

    start_time = time.time()
    print('start time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    intensity = get_intensity(df_HW_final= df,model_name=model_name)
    end_time =time.time()
    take_time3 = time.time() - start_time
    print('cal_intensity end time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    print('it took ', round(take_time3,2), ' seconds to calculate intensity.')
    print()

    take_all_time =  take_time1 + take_time2 + take_time3
    print('total time: ', round(take_all_time,2), ' seconds to calculate all metrics.')
    if is_merge:
        print()
        merge_start_time = time.time()
        df_final = pd.merge(frequency,duration,on=['lat','lon']).rename(columns={model_name+'_x':'frequency',model_name+'_y':'duration'})
        df_final = pd.merge(df_final,intensity,on=['lat','lon']).rename(columns={model_name:'intensity'})
        df_final = pd.merge(df_final,df,on=['lat','lon']).rename(columns={'duration_x':'total_dur',
                                                    'duration_y':'single_dur'}).reset_index().sort_values(by=['lat','lon','datetime']).drop(['HW'],axis=1)
        merge_end_time = time.time()
        df_final = df_final[['lat','lon','datetime','mean','quant','frequency','total_dur','single_dur','intensity']].rename(columns={'mean':var_name})
        print('merge time: ', round(merge_end_time-merge_start_time,2), ' seconds to merge all metrics.')
        merge_time = merge_end_time-merge_start_time
        print('it take ', round(take_all_time + merge_time,2), ' seconds to calculate and merge all metrics.')
        print('-'*50)
        print()
        return df_final,[frequency,duration,intensity],take_all_time + merge_time,[take_time1,take_time2,take_time3]
    return [frequency,duration,intensity],take_all_time,[take_time1,take_time2,take_time3]

    





#------------------------------ODPS basic function-----------------------------------
def init_ODPS():
    """
    作用:初始化ODPS
    输出:ODPS对象
    """
    project= ''
    ak=''
    sk=''
    endpoint=''
    o = ODPS(ak,sk,project,endpoint=endpoint)
    now_time = time.time()
    print('time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_time)))
    return o

def show_instance_statues(o,need_stop = False):    
    """
    作用:显示当前ODPS的instance的状态
    输入:
        o:ODPS对象
        need_stop:是否需要停止所有的instance,默认为False
    """
    cnt = 0
    terminate_num = 0
    running_num = 0
    now_time = time.time()
    for instance in o.list_instances():
        cnt += 1
        if instance.status == Instance.Status.TERMINATED:
            terminate_num += 1
        else:
            print(instance.id,'  ',instance.status)
        if instance.status == Instance.Status.RUNNING:
            running_num += 1
            if need_stop:
                o.stop_instance(instance.id)
    print('-'*50)
    print('time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_time)))
    print()
    print('-'*50)
    print('time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_time)))
    print('total instance num: ', cnt)
    print('terminate instance num: ', terminate_num)
    print('running instance num: ', running_num)


def show_all_tables(o):
    """
    作用:显示当前ODPS的所有表
    """
    now_time = time.time()
    print('start time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_time)))
    for table in o.list_tables():
        print(table.name)



def show_get_table_info(o,table_name):
    """
    作用:显示当前ODPS的某个表的信息
    输入:
        o:ODPS对象
        table_name:表名
    输出:表的信息
    """
    now_time = time.time()
    print('start time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_time)))
    print('-'*50)
    t = o.get_table(table_name)
    print(t)
    print('size: ' + str(t.size)) 
    print('-'*50)
    return t

def get_instance_dur(o,instance_id,show_type = 'second'):
    """
    作用:显示当前ODPS的某个instance的运行时间
    输入:
        o:ODPS对象
        instance_id:instance的id
        show_type:显示的时间类型,默认为'second',可选为'second','minute','hour'
    输出:instance的运行时间
    """

    ins = o.get_instance(instance_id)
    print('instance id: ', ins.id,end='     ')
    print('start time: ', ins.start_time,end='     ')
    print('end time: ', ins.end_time)
    diff = ins.end_time - ins.start_time
    seconds = diff.total_seconds()
    print('duration: ', diff ,'     dur_seconds: ', seconds)


def plot_instance_dur(o,all_num = 100,show_number = 100,time_list = None,show_type = 'second',label_interval = 10):
    """
    作用:显示当前ODPS的某个区间(最新跑的all_num个instance中,随机抽取show_num个instance,进行绘图展示所用时间)的instance的运行时间的图像
    输入:
        o:ODPS对象
        all_num:instance的总数,默认为100,也即最近100个instance
        show_number:显示的instance的数量,默认为100,也即最近100个instance
        time_list:instance的时间列表,默认为None,如果为None,则会随机选择show_number个instance
        show_type:显示的时间类型,默认为'second',可选为'second','minute','hour'
        label_interval:显示的x轴的label的间隔,默认为10
    显示:instance的运行时间的图像
    """
    if time_list is None:
        time_list = []
        total_instance = [instance for instance in o.list_instances()]
        all_instance = total_instance[-1*all_num:]
        random_sample = random.sample(all_instance,show_number)
        display_instance = sorted(random_sample,key = lambda x:x.start_time)
        for instance in display_instance:
            time_list.append([instance.start_time,instance.end_time])
    # get the start time and end time
    start_times = [item[0] for item in time_list]
    end_times = [item[1] for item in time_list]

   ## get the time intervals in different type
    if show_type == 'second':
        time_intervals = [(end - start).seconds for start, end in time_list]
        
    elif show_type == 'minute':
        time_intervals = [(end - start).seconds / 60 for start, end in time_list]

    elif show_type == 'hour':
        time_intervals = [(end - start).seconds / 3600 for start, end in time_list]
    
    avg_time = sum(time_intervals) / len(time_intervals)

    total_time = sum(time_intervals)*  all_num / show_number
    # transfer the time to hour
    if show_type == 'second':
        total_time = total_time / 3600
        avg_time = avg_time / 3600
        used_time = max(time_intervals) / 3600
    elif show_type == 'minute':
        total_time = total_time / 60
        used_time = max(time_intervals) / 60
        avg_time = avg_time / 60
    # limit the float number in 2
    print('total instance: ', all_num,'show instance: ', show_number)
    print(f'total time if serial: {total_time:.2f}', 'hour')
    print(f'average time per instance: {avg_time:.2f}', 'hour')
    print(f'used time: {used_time:.2f}', 'hour')
    plt.figure(figsize=(20, 10))

    # plot the figure
    if show_type == 'second':
        plt.bar(range(len(time_list)), time_intervals, bottom=[start.second for start in start_times], width=0.5, color='skyblue')
    elif show_type == 'minute':
        plt.bar(range(len(time_list)), time_intervals, bottom=[start.minute for start in start_times], width=0.5, color='skyblue')
    elif show_type == 'hour':
        plt.bar(range(len(time_list)), time_intervals, bottom=[start.hour for start in start_times], width=0.5, color='skyblue')

    # set the label of the figure
    plt.ylabel('Time /'+show_type)
    plt.xlabel('Elements')
    plt.title('Time Intervals of Elements')
    # label the x axis with every ten elements
    plt.xticks(range(0, len(time_list), label_interval), [f'{i+1}' for i in range(0, len(time_list), label_interval)])
    # plot average time
    if show_type == 'second':
        avg_time = avg_time *3600
    elif show_type == 'minute':
        avg_time = avg_time * 60
    
    plt.plot([0, len(time_list)], [avg_time, avg_time], color='red', linewidth=2.0, linestyle='--')
    # label the average time in right position
    plt.legend(['Average Time', 'Time Intervals'], loc='upper right')
    # show the average time line corresponding to the y axis
    plt.text(0, avg_time + 0.5, f'{avg_time:.2f}')
    # show the figure
    plt.show()

    



def init_table_info():
    """
    作用:初始化ODPS的表的信息
    输出:表的columns和partitions
    """
    columns = [
            #Column(name='member_id', type='int', comment='member id of cesm-lens1'), 
            Column(name='lat', type='double', comment='latitude, unit: degree'),
            Column(name='lon', type='double', comment='longitudue,unit: degree'),
            Column(name='time_list', type='TIMESTAMP', comment='date'),
            Column(name='climate_index', type='double', comment='index of cliamte (e.g., temperature, precipitation), unit: (e.g.,C, M/S)'),
            Column(name='quant', type='double', comment='quantile value of climate index at a specific percentile, unit: (e.g.,C, M/S)'),
            Column(name='frequence', type='int', comment='counts of the number cliamte extreme events in the seleacted peroid, unit: counts'),
            Column(name='mean_climate_index', type='double', comment='mean of the cliamte events in the seleacted peroid, unit: (e.g.,C, M/S)'),
            Column(name='group_id', type='double', comment=''),
            Column(name='single_event_dur', type='int', comment='single event duration, unit: days'),
            Column(name='total_duration', type='int', comment='total duration of climate extreme events in the seleacted peroid, unit: days'),
            #Column(name='percentile', type='int', comment='the percentile for calculating the quantile'),
            #Column(name='event_kind', type='string', comment='kind of climate extreme events'),
            ]
    partitions = [
        #Partition(name='percentile_period', type='string', comment='the period of the quantile'), 
                Partition(name='percentile', type='int', comment='the percentile for calculating the quantile'),
                Partition(name='member_id', type='int', comment='member id of cesm-lens1'),
                  Partition(name='event_kind', type='string', comment='the partition')]
    return columns,partitions

def create_table(o,table_name,columns,partitions=None,comment='(without reference date)This table is created for the storation of the climate extremes calculated from CESM-LENS1.'):
    """
    作用:创建ODPS的表
    输入:
        o:ODPS对象
        table_name:创建的表名
        columns:表的结构,columns
        partitions:表的partitions,默认为None
        comment:表的注释,默认为'(without reference date)This table is created for the storation of the climate extremes calculated from CESM-LENS1.'
    """
    if partitions is None:
        schema = TableSchema(columns = columns)
    else:
        schema = TableSchema(columns=columns,partitions=partitions)
    table_yjj = o.create_table(table_name, 
                            schema,                            
                            if_not_exists=True, 
                            comment=comment)
    now_time = time.time()
    print('start time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_time)))
    print('table %s is created successfully!' % table_name)


def drop_table(o,table_name):
    """
    作用:删除ODPS的表
    输入:
        o:ODPS对象
        table_name:删除的表名
    """
    print('table info: ')
    t = show_get_table_info(o,table_name)
    t.drop()
    print()
    print('table %s is dropped successfully!' % table_name)



#------------------------------sql2ODPS main function-----------------------------------


def read_sql(path,input_stream, output_filename='output.txt'):
    """
    作用:使用文本编辑器的方法,读取sql脚本,并将其中的超参数进行替换,转换成为pyodps合适的输入
    输入:
        path:sql脚本的路径
        input_stream:sql脚本中的参数,字典形式
        output_filename:输出的sql脚本的路径,默认为'output.txt',用于调试代码查看结果
    """

    # 此函数用于将sql脚本的参数进行替换，如阈值、时间、换行符等，转换成为pyodps合适的输入
    with open(path, 'r') as file:
        script = file.read()
        script = re.sub(r'--.*','',script) 
        for i in input_stream:
            # 由于sql脚本中的参数是以@开头的，因此需要加上@
            input_sc = '@' + i
            # 由于sql脚本中的参数是字符串，因此需要加上引号
            if i in ["event_kind" , "percentile_period"]:
                script = script.replace(input_sc, "\'"+input_stream[i]+"\'")
            else:
                script = script.replace(input_sc, input_stream[i])

        with open(output_filename, 'w') as file:
                file.write(script)
        return script


def get_config(json_path):
    """
    作用:读取json文件,获取用户输入的参数
    输入:
        json_path:json文件的路径
    输出:用户输入的参数,字典形式
    """
    # get the args of user from the config.json file
    with open(json_path, 'r') as f:
        config = json.load(f)

    # create ArgumentParser object
    parser = argparse.ArgumentParser()

    # add the args
    parser.add_argument('--var_col_name', help='replace the column name with your own column name', default=config.get('var_col_name'))
    parser.add_argument('--drop_nan', help='whether to drop the nan value', default=config.get('drop_nan'),choices=['TRUE','FALSE'])
    parser.add_argument('--select_positive',help='whether to select the positive value', default=config.get('select_positive'),choices=['TRUE','FALSE'])
    parser.add_argument('--lat_col_name', help='replace the lat column name with your own column name', default=config.get('lat_col_name'))
    parser.add_argument('--lon_col_name', help='replace the lon column name with your own column name', default=config.get('lon_col_name'))
    parser.add_argument('--time_col_name', help='replace the time column name with your own column name', default=config.get('time_col_name'))
    parser.add_argument('--start_year', help='Input the start year for calculating the events', default=config.get('start_year'))
    parser.add_argument('--end_year', help='Input the end year for calculating the events', default=config.get('end_year'))
    parser.add_argument('--start_month', help='Input the start month for calculating the events', default=config.get('start_month'))
    parser.add_argument('--end_month', help='Input the end month for calculating the events', default=config.get('end_month'))
    parser.add_argument('--SELECT_ALL_TIME', help='''When you want to select all the time, please set it as TRUE, if FALSE, 
                                                please input the start time and end time''', default=config.get('SELECT_ALL_TIME'),choices=['TRUE','FALSE'])
    parser.add_argument('--set_quantile', help='''Input the absolute quantile for calculating the events, 
                                                if not absolute, please set it as NULL''', default=config.get('set_quantile'))
    parser.add_argument('--quant', help='''Input the quantile for calculating the events,
                                    if you want to calculate the absolute quantile, then set set_quantile instead,which will cover the quant,
                                    if you want to select quantifile as a list,then input a list like [start,end,step]
                                    ''', default=config.get('quant'))
    
    parser.add_argument('--duration_threshold', help='Input the duration threshold for calculating the events', default=config.get('duration_threshold'))
    parser.add_argument('--member_id', help='''Input the member_id for calculating the events, 
                                             if if you want to select member_id as a list,then input a list like [[start,end,step],[start,end,step]],
                                             if not exist, please set it as NULL''', default=config.get('member_id'))
    
    parser.add_argument('--SQL_path', help='Input the SQL script path', default=config.get('SQL_path'))
    parser.add_argument('--output_filename', help='the intermediate output SQL filename in text', default=config.get('output_file_name'))
    parser.add_argument('--tablename', help='Input the table form your project', default=config.get('tablename'))
    parser.add_argument('--output_table', help='Input the output table name', default=config.get('output_table'))



    # parse the args(解析命令行参数)
    args = parser.parse_args(args=[])
    # convert the args to dict
    args_dict = vars(args)
    # print the args to the screen for user to check
    for i,arg in enumerate(args_dict):
        # 每次结果占14个字符的宽度，不足的用空格补齐
        # print("{:<16}:{:<10}".format(arg,getattr(args, arg)),end = '        ' if (i+1)%3  else '\n')
        value = getattr(args, arg)
        print(f"{arg:<18}: {str(value):<30}", end='' if (i + 1) % 3 else '\n')
        # print(f"{arg}: {getattr(args, arg)}",end = '        ' if (i+1)%3  else '\n')
    return args_dict


def odps_exec(json_path = 'config.json',loop_order = ['member_id','percentile','var'],need_answer = True,need_loop = True,need_modify = False,modify_dict = {},merge_mode = False):
    """
    作用:执行ODPS的sql脚本,运行时,首先会读取json文件,获取用户输入的参数并显示,让用户检查是否正确,如果正确,则输入Y,执行sql脚本,
                                                    如果不正确,用户需要输入"N",则会让用户修改参数,修改完成后,按下任意键继续,会重新询问用户是否正确,直到用户输入Y为止
                                                    如果用户输入Q,则会退出程序
    输入:
        json_path:json文件的路径,默认为'config.json'
        loop_order:需要循环执行的参数,默认为['member_id','percentile','var'],若有些数据集不需要循环member_id,则可以将member_id,则可以修改为['percentile','var']
        need_answer:是否需要用户确认输入的参数,默认为True,如果为False,则不会询问用户是否正确,直接执行sql脚本
        need_loop:是否需要循环执行sql脚本,默认为True,如果为False,则只执行一次sql脚本
        need_modify:是否需要修改输入的参数,默认为False,如果为True,则会读取modify_dict,并将其中的参数替换到json文件中,然后执行sql脚本
        modify_dict:需要修改的参数,字典形式,默认为{}
        merge_mode:是否需要合并结果,默认为False,如果为True,则会启用合并表的模式,读取06-15年quant,作为后续计算的quantile_avail
    """

    # this function is used to execute the sql script in the odps
    count = 0

    while True:
        print()
        print('-'*25+'-'*25)
        if need_modify:
            input_stream = modify_dict
            print('modify_dict: ', modify_dict)
        else:
            # get the config from the config.json file and print it to the screen for user to check
            input_stream = get_config(json_path)
        # ask the user whether the config is correct
        print()
        print('Is that correct?[Y/N] or [Q] to quit!')
        print()
        if need_answer:
            answer = input()
        else:
            answer = 'Y'
        # if user want to quit, then quit the program
        if answer in ['Q','q','Quit','quit','QUIT']:
            print('Succeed in quitting the program')
            break
        # if user think the config is incorrect, then continue the loop
        elif answer in ['N','n','No','no','NO']:
            print('Please check the input!')
            # after the user modify the cofig then ask the user to press any key to continue
            print('press any key to continue')
            input()
            continue
        else:
            # if the user think the config is correct, then start to execute the sql script
            print('-'*15+'Start to execute the %dnd sql script!' % (count+1)+'-'*15)
            now_time = time.time()
            print('start time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_time)))

            # get the args from the config.json file which need to be looped
            ## member_id_raw is a 1D/2D list or a int
            member_id_raw = input_stream['member_id'] if (input_stream['member_id'] != 'NULL') else 0
            member_id = []
            # if the member_id_raw is list
            if isinstance(member_id_raw, list):
                # the member_id_raw is a 1D list,convert it to 2D list
                if not isinstance(member_id_raw[0], list):
                    member_id_raw = [member_id_raw]
                # convert the member_id_raw to a 1D list
                for i in range(len(member_id_raw)):
                    member_id += list(np.arange(member_id_raw[i][0],member_id_raw[i][1],member_id_raw[i][2]))
            # if the member_id_raw is int
            else:
                member_id = [member_id_raw]
            ## percentile_raw is a list or a float
            percentile_raw = input_stream['set_quantile'] if (input_stream['set_quantile'] != 'NULL') else input_stream['quant']
            percentile = []
            # if the percentile_raw is list
            if isinstance(percentile_raw, list):
                # the percentile_raw is a 1D list,convert it to 2D list
                if not isinstance(percentile_raw[0], list):
                    percentile_raw = [percentile_raw]
                # convert the percentile_raw to a 1D list
                for i in range(len(percentile_raw)):
                    percentile += list(np.arange(percentile_raw[i][0],percentile_raw[i][1],percentile_raw[i][2]))
            # if the percentile_raw is float
            else:
                percentile = [percentile_raw]

            # get the var_list from the config.json file which need to be looped,convert it to a list
            var_list = input_stream['var_col_name'] if isinstance(input_stream['var_col_name'], list) else [input_stream['var_col_name']]
            # get the sql script path from the config.json file which need to be saved as intermediate SQL file
            sql_path = input_stream['SQL_path']
            # get the output filename from the config.json file which need to be saved as intermediate SQL file
            output_filename = input_stream['output_filename']
            # if you want to select all the time, then set the SELECT_ALL_TIME as TRUE
            input_stream['SELECT_ALL_TIME'] = '1' if (input_stream['SELECT_ALL_TIME'] in ["TRUE","1"]) else '0'
            # if you want to drop the nan value, then set the drop_nan as TRUE
            input_stream['drop_nan'] = '1' if (input_stream['drop_nan'] in ["TRUE","1"]) else '0'
            # if you want to select the positive value, then set the select_positive as TRUE
            input_stream['select_positive'] = '1' if (input_stream['select_positive'] in ["TRUE","1"]) else '0'
            print(f'member_id:(total:{len(member_id)}) \n', member_id)
            print(f'percentile:(total:{len(percentile)}) \n', percentile)
            print(f'var_list:(total:{len(var_list)}) \n', var_list)
            print('drop_nan: \n', input_stream['drop_nan'])
            print('-'*50)
            print('total task: ', len(member_id)*len(percentile)*len(var_list))
            print('start time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_time)))
            print('-'*50)
            print()
            # -----------------------以下注释内容为实现以用户输入的参数的顺序的为循环顺序的代码-----------------------------------
            # index_dic = {}
            # no_member = False
            # # 将用户输入的参数的顺序转换成为索引的顺序
            # if "member_id" in loop_order:
            #     member_id_index = loop_order.index("member_id")
            #     index_dic[member_id_index] = member_id
            # else:
            #     # 由于member_id 不在循环中，因此需要判断member_id是否为NULL
            #     assert input_stream['member_id'] == 'NULL', 'member_id is not in your loop_order and member_id is not NULL, please check the config.json file'
            #     # 如果member_id不在循环中，则将member_id设置为0，其顺序为2
            #     index_dic[0] = member_id
            #     no_member = True
            # if "percentile" in loop_order:
            #     percentile_index = loop_order.index("percentile")
            #     if no_member:
            #         percentile_index = percentile_index + 1
            #     index_dic[percentile_index] = percentile
            # if "var" in loop_order:
            #     var_index = loop_order.index("var")
            #     if no_member:
            #         var_index = var_index + 1
            #     index_dic[var_index] = var_list
            # # 将用户顺序，读取变量循环的顺序
            # order_list = [index_dic[i] for i in range(len(index_dic))]
            # x = order_list[0]
            # y = order_list[1]
            # z = order_list[2]
            # x_mesh, y_mesh, z_mesh = np.meshgrid(x,y,z, indexing='ij')
            # combinations = np.column_stack((x_mesh.ravel(), y_mesh.ravel(), z_mesh.ravel()))  

            # 强制要求用户输入的参数的顺序为['member_id','percentile','var']或['percentile','var']（没有member_id的情况）
            assert loop_order in [['member_id','percentile','var'],['percentile','var'],['var']], 'loop_order should be in [[\'member_id\',\'percentile\',\'var\'],[\'percentile\',\'var\'],[\'var\']]'
            if loop_order == ['percentile','var']:
                assert input_stream['member_id'] == 'NULL', 'member_id is not in your loop_order and member_id is not NULL, please check the config.json file'
            x = member_id
            y = percentile
            z = var_list
            # 网格化，使得对于每个i,j,k，都可以得到一个组合，共i*j*k个组合
            x_mesh, y_mesh, z_mesh = np.meshgrid(x,y,z, indexing='ij')
            combinations = np.column_stack((x_mesh.ravel(), y_mesh.ravel(), z_mesh.ravel()))

            # add the module of the visualizing with the processing bar
            processing_bar = tqdm(combinations,leave= True,position=0)
            # for i in member_processing_bar:
            #     # add the module of the visualizing with the processing bar,to ensure the user know which member is processing
            #     percentile_processing_bar = tqdm(percentile,leave= True,position=1)
            #     for j in percentile_processing_bar:
            #         for k in var_list:
                        # print('-'*50)
                        # print('member_id: ', i, 'percentile: ', j, 'var: ', k)
                        # print('-'*50)
            for loop_set in processing_bar:
                i,j,k = loop_set
                input_stream['member_id'] = str(i)
                input_stream['quant'] = str(j)
                input_stream['var_col_name'] = str(k)
                # 若var时温度，则不需要选择正值（由于温度的单位为开尔文，>0）
                if k == 'trefhtmx':
                    input_stream['select_positive'] = "0"
                # 若var时降水，则需要选择正值(由于某些地区降水=0，相当于drop降水=0的值)
                elif k == 'prect':
                    input_stream['select_positive'] = "1"
                # 若member_id为0，也即在json中其值=NULL,则设置all_member为1，否则为0
                if i == 0:
                    input_stream['all_member'] = '1'
                else:
                    input_stream['all_member'] = '0'
                processing_bar.set_description("Processing member: %s,Processing percentile: %s , var is %s" % (str(i) if 
                                                (i != 0) else 'NULL',str(j*100)+'%'  
                                            if (input_stream['set_quantile'] == 'NULL')else str(j)+'℃',str(k)))
                # percentile_processing_bar.set_description("Processing percentile: %s , var is %s " % (str(j*100)+'%'
                if merge_mode:
                    input_stream['event_kind'] = str(k)
                    input_stream['percentile_period'] = str(input_stream['start_year'])+'-'+input_stream['start_month'] + \
                                            '-'+str(input_stream['end_year'])+'-'+input_stream['end_month']
                # member_processing_bar.set_description("Processing member: %s" % (str(i) if 
                #                                 (i != 0) else 'NULL'))
                # percentile_processing_bar.set_description("Processing percentile: %s , var is %s " % (str(j*100)+'%'  
                #                             if (input_stream['set_quantile'] == 'NULL')else str(j)+'℃',str(k)))

                project= ''
                ak=''
                sk=''
                endpoint=''
                o = ODPS(ak,sk,project,endpoint=endpoint)
                proj = o.get_project()
                result = read_sql(sql_path, input_stream, output_filename)
                o.run_sql(result)
                

        count += 1
        if not need_loop:
            break
        print('-'*15+'Finish to execute the %dnd sql script!' % (count+1)+'-'*15)
        now_time = time.time()
        print('end time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_time)))


#################################################################################
# def odps_exec_ass1(json_path = 'config.json',need_answer = True,need_loop = True,need_modify = False,modify_dict = {},merge_mode = False):
#     # this function is used to execute the sql script in the odps
#     count = 0

#     while True:
#         print()
#         print('-'*25+'-'*25)
#         if need_modify:
#             input_stream = modify_dict
#             print('modify_dict: ', modify_dict)
#         else:
#             # get the config from the config.json file and print it to the screen for user to check
#             input_stream = get_config(json_path)
#         # ask the user whether the config is correct
#         print()
#         print('Is that correct?[Y/N] or [Q] to quit!')
#         print()
#         if need_answer:
#             answer = input()
#         else:
#             answer = 'Y'
#         # if user want to quit, then quit the program
#         if answer in ['Q','q','Quit','quit','QUIT']:
#             print('Succeed in quitting the program')
#             break
#         # if user think the config is incorrect, then continue the loop
#         elif answer in ['N','n','No','no','NO']:
#             print('Please check the input!')
#             # after the user modify the cofig then ask the user to press any key to continue
#             print('press any key to continue')
#             input()
#             continue
#         else:
#             # if the user think the config is correct, then start to execute the sql script
#             print('-'*15+'Start to execute the %dnd sql script!' % (count+1)+'-'*15)
#             now_time = time.time()
#             print('start time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_time)))

#             # get the args from the config.json file which need to be looped
#             ## member_id_raw is a 1D/2D list or a int
#             member_id_raw = input_stream['member_id'] if (input_stream['member_id'] != 'NULL') else 0
#             member_id = []
#             if isinstance(member_id_raw, list):
#                 # the member_id_raw is a 1D list
#                 if not isinstance(member_id_raw[0], list):
#                     member_id_raw = [member_id_raw]
#                 for i in range(len(member_id_raw)):
#                     member_id += list(np.arange(member_id_raw[i][0],member_id_raw[i][1],member_id_raw[i][2]))
#             else:
#                 member_id = [member_id_raw]
#             ## percentile_raw is a list or a float
#             percentile_raw = input_stream['set_quantile'] if (input_stream['set_quantile'] != 'NULL') else input_stream['quant']
#             percentile = []
#             if isinstance(percentile_raw, list):
#                 if not isinstance(percentile_raw[0], list):
#                     percentile_raw = [percentile_raw]
#                 for i in range(len(percentile_raw)):
#                     percentile += list(np.arange(percentile_raw[i][0],percentile_raw[i][1],percentile_raw[i][2]))
#             else:
#                 percentile = [percentile_raw]

#             var_list = input_stream['var_col_name'] if isinstance(input_stream['var_col_name'], list) else [input_stream['var_col_name']]
#             # get the sql script path from the config.json file which need to be saved as intermediate SQL file
#             sql_path = input_stream['SQL_path']
#             # get the output filename from the config.json file which need to be saved as intermediate SQL file
#             output_filename = input_stream['output_filename']
#             # if you want to select all the time, then set the SELECT_ALL_TIME as TRUE
#             input_stream['SELECT_ALL_TIME'] = '1' if (input_stream['SELECT_ALL_TIME'] in ["TRUE","1"]) else '0'
#             # if you want to drop the nan value, then set the drop_nan as TRUE
#             input_stream['drop_nan'] = '1' if (input_stream['drop_nan'] in ["TRUE","1"]) else '0'
#             # if you want to select the positive value, then set the select_positive as TRUE
#             input_stream['select_positive'] = '1' if (input_stream['select_positive'] in ["TRUE","1"]) else '0'
#             print(f'member_id:(total:{len(member_id)}) \n', member_id)
#             print(f'percentile:(total:{len(percentile)}) \n', percentile)
#             print(f'var_list:(total:{len(var_list)}) \n', var_list)
#             print('drop_nan: \n', input_stream['drop_nan'])
#             print('-'*50)
#             print('total task: ', len(member_id)*len(percentile)*len(var_list))
#             print('start time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_time)))
#             print('-'*50)
#             print()
#             # add the module of the visualizing with the processing bar
#             member_processing_bar = tqdm(member_id,leave=True,position=0) if (input_stream['member_id'] != 'NULL') else tqdm([0],leave=True,position=0)
#             for i in member_processing_bar:
#                 # add the module of the visualizing with the processing bar,to ensure the user know which member is processing
#                 percentile_processing_bar = tqdm(percentile,leave= True,position=1)
#                 for j in percentile_processing_bar:
#                     for k in var_list:
#                         # print('-'*50)
#                         # print('member_id: ', i, 'percentile: ', j, 'var: ', k)
#                         # print('-'*50)
#                         input_stream['member_id'] = str(i)
#                         input_stream['quant'] = str(j)
#                         input_stream['var_col_name'] = str(k)
#                         input_stream['tablename'] = 'cesm_' + str(k)

#                         if k == 'trefhtmx':
#                             input_stream['select_positive'] = "0"
#                         elif k == 'prect':
#                             input_stream['select_positive'] = "1"

#                         input_stream['event_kind'] = str(k)

#                         input_stream['percentile_int'] = str(int(j*100))

#                         member_processing_bar.set_description("Processing member: %s" % (str(i) if 
#                                                         (i != 0) else 'NULL'))
#                         percentile_processing_bar.set_description("Processing percentile: %s , var is %s " % (str(j*100)+'%'  
#                                                     if (input_stream['set_quantile'] == 'NULL')else str(j)+'℃',str(k)))

#                         project= ''
#                         ak=''
#                         sk=''
#                         endpoint=''
#                         o = ODPS(ak,sk,project,endpoint=endpoint)
#                         proj = o.get_project()
#                         result = read_sql(sql_path, input_stream, output_filename)
#                         o.run_sql(result)
#         count += 1
#         if not need_loop:
#             break
