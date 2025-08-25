import pandas as pd

def create_normal_lags(df: pd.DataFrame, lags: int = 8):
    df = df.copy()
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['ActualPrice'].shift(i)
    df.dropna(inplace=True)
    return df

def create_weekly_lags(df):
    df_copy = df.copy()
    df_copy['time_id'] = df_copy.groupby(df_copy.index.date).cumcount() + 1
    df_copy['lag_date'] = df_copy.index.normalize() - pd.Timedelta(weeks=1)
    lagged_data = df_copy[['ActualPrice', 'time_id']].copy()
    lagged_data.index = df_copy['lag_date']
    lagged_data.sort_index(inplace=True)

    for i in range(1, 8):
        lagged_slice = lagged_data[lagged_data['time_id'] == i].drop(columns=['time_id'])
        lagged_slice.rename(columns={'ActualPrice': f'week_lag_{i}'}, inplace=True)
        df_copy = pd.merge_asof(df_copy, lagged_slice, left_on='lag_date', right_index=True, direction='nearest')
    df_copy.drop(columns=['time_id', 'lag_date'], inplace=True)
    df_copy.dropna(inplace=True)
    
    return df_copy

def create_monthly_lags(df: pd.DataFrame):
    df_copy = df.copy()
    df_copy['time_id'] = df_copy.groupby(df_copy.index.date).cumcount() + 1
    df_copy['lag_date'] = df_copy.index.normalize() - pd.DateOffset(months=1)
    
    lagged_data = df_copy[['ActualPrice', 'time_id']].copy()
    lagged_data.index = df_copy['lag_date']
    lagged_data.sort_index(inplace=True)

    for i in range(1, 8):
        lagged_slice = lagged_data[lagged_data['time_id'] == i].drop(columns=['time_id'])
        lagged_slice.rename(columns={'ActualPrice': f'month_lag_{i}'}, inplace=True)
        df_copy = pd.merge_asof(df_copy, lagged_slice, left_on='lag_date', right_index=True, direction='nearest')
    df_copy.drop(columns=['time_id', 'lag_date'], inplace=True)
    df_copy.dropna(inplace=True)
    
    return df_copy

def create_yearly_lag(df):
    df_copy = df.copy()
    lag_date = df_copy.index - pd.DateOffset(years=1)    
    lagged_df = pd.DataFrame(df_copy['ActualPrice'].values, index=lag_date, columns=['year_lag'])    
    lagged_df.sort_index(inplace=True)    
    df_copy = pd.merge_asof(df_copy, lagged_df, left_index=True, right_index=True, direction='nearest')
    return df_copy

import pandas as pd
import os

def update_all_advanced_lags(df: pd.DataFrame, lagged_path: str):
    """
    Incrementally update lagged data stored on disk.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe with 'ActualPrice' column and DatetimeIndex.
    lagged_path : str
        Path to existing lagged_df file (CSV).

    Returns
    -------
    pd.DataFrame
        Updated lagged_df containing lag features for all data in df.
    """

    # Load lagged_df from disk if exists, else create empty
    if os.path.exists(lagged_path):
        lagged_df = pd.read_csv(lagged_path, index_col=0, parse_dates=True)
        lagged_df.sort_index(inplace=True)
    else:
        lagged_df = pd.DataFrame()

    # Ensure df sorted
    df = df.sort_index()

    # If lagged_df already up to date
    if not lagged_df.empty and df.index[-1] == lagged_df.index[-1]:
        return lagged_df

    # Find missing timestamps
    if lagged_df.empty:
        missing_idx = df.index
    else:
        missing_idx = df.index.difference(lagged_df.index)

    if missing_idx.empty:
        return lagged_df

    # Extract only missing data
    new_data = df.loc[missing_idx].sort_index()

    # Merge for lag calculation context
    if lagged_df.empty:
        combined_df = df.copy()
    else:
        combined_df = pd.concat([lagged_df[['ActualPrice']], new_data])
    combined_df = combined_df.sort_index()

    # Need enough previous rows for lags
    window_size = max(8, 7, 7, 1)  # largest lag depth
    start_pos = max(0, combined_df.shape[0] - (len(new_data) + window_size))
    recompute_df = combined_df.iloc[start_pos:].copy()

    # Apply lag functions
    recompute_df = create_normal_lags(recompute_df)
    recompute_df = create_weekly_lags(recompute_df)
    recompute_df = create_monthly_lags(recompute_df)
    recompute_df = create_yearly_lag(recompute_df)
    recompute_df.dropna(inplace=True)

    # Keep only newly computed missing rows
    recompute_df = recompute_df.loc[missing_idx.intersection(recompute_df.index)]

    # Append new rows to lagged_df
    lagged_df = pd.concat([lagged_df, recompute_df]).sort_index()

    # Save updated lagged_df back to disk
    lagged_df.to_csv(lagged_path)

    return lagged_df


def create_all_advanced_lags(df):
    df_with_lags = df.copy()
    df_with_lags = create_normal_lags(df_with_lags)
    df_with_lags = create_weekly_lags(df_with_lags)
    df_with_lags = create_monthly_lags(df_with_lags)
    df_with_lags = create_yearly_lag(df_with_lags)
    df_with_lags.dropna(inplace=True)    
    return df_with_lags