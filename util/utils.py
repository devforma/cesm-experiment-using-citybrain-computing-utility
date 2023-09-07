import pandas as pd
import numpy as np
import time
import gc


# load data csv
def load_df(file_path):
    print("Start to load csv", file_path)
    time_point = time.time()
    df_raw = pd.read_csv(file_path)
    print("It takes", time.time()-time_point, "to load csv")
    return df_raw

# function for getting mean value of each grid for each day 
def get_mean(df_raw, model_type):
    # calculate mean
    if model_type == "cesm":
        df_raw_mean = df_raw.set_index(["lat","lon","time"]).mean(axis=1).reset_index().rename(columns={0:"mean"})
        # convert date to datetime
        df_raw_mean["datetime"]=(pd.to_datetime(df_raw_mean["time"],errors="coerce"))
    else:
        # For the latest version, I used "time" rather than date for CMIP
        #df_raw_mean = df_raw.set_index(["lat","lon","date"]).mean(axis=1).reset_index().rename(columns={0:"mean"})
        # convert date to datetime
        #df_raw_mean["datetime"]=(pd.to_datetime(df_raw_mean["date"]))
        
        df_raw_mean = df_raw.set_index(["lat","lon","time"]).mean(axis=1).reset_index().rename(columns={0:"mean"})
        # convert date to datetime
        df_raw_mean["datetime"]=(pd.to_datetime(df_raw_mean["time"],errors="coerce"))
        
        
    # sort datetime
    df_raw_date_sort = df_raw_mean[["lat","lon","datetime","mean"]].sort_values(["lat","lon","datetime"]).reset_index(drop=True)
    return df_raw_date_sort

# function for calculating single model quantile value
def single_model_quant(df, set_quantile, variable):
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
    df_HW = df[df["HW"]==0].reset_index(drop=True)
    dt = df_HW['datetime']
    day = pd.Timedelta('1d')
    in_block = ((dt - dt.shift(-1)).abs() == day) | (dt.diff() == day)
    breaks = dt.diff() != day
    df_HW["group_id"] = breaks.cumsum()
    return df_HW

# get heat waves dataframe for 2006
def get_heat_waves_df(df, set_quantile, duration_threshold, model_type, quantile_avail):
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
    df_HW_final_group = df_HW_final["group_id"].drop_duplicates().reset_index()
    df_HW_final_group["num"] = 1
    df_HW_freq=df_HW_final_group.groupby(["lat","lon"]).sum()["num"]
    df_HW_freq = pd.DataFrame(df_HW_freq).reset_index()
    df_HW_freq[model_name] =  df_HW_freq["num"]/10
    df_HW_freq_final = df_HW_freq[["lat","lon",model_name]].set_index(["lat","lon"])

    return df_HW_freq_final

# get duration from the output of the "get_heat_waves_df"
def get_duration(df_HW_final,model_name):
    df_HW_duration = df_HW_final[["group_id","duration"]].drop_duplicates()["duration"].reset_index().groupby(["lat","lon"]).sum()["duration"]
    df_HW_duration = pd.DataFrame(df_HW_duration).reset_index()
    df_HW_duration[model_name] = df_HW_duration["duration"]/10
    df_HW_duration_final = df_HW_duration[["lat","lon",model_name]].set_index(["lat","lon"])

    return df_HW_duration_final

# get intensity from the output of the "get_heat_waves_df"
def get_intensity(df_HW_final,model_name):
    df_HW_intensity = pd.DataFrame(df_HW_final.reset_index().groupby(["lat","lon"]).mean()["mean"]).reset_index()
    df_HW_intensity[model_name] = df_HW_intensity["mean"]
    df_HW_intensity_final = df_HW_intensity[["lat","lon",model_name]].set_index(["lat","lon"])

    return df_HW_intensity_final

