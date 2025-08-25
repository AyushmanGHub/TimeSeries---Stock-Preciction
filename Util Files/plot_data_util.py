import pandas as pd
import plotly.graph_objects as go
import math
import numpy as np
import os

def plot_train_and_prediction(data: pd.DataFrame, ticker: str, n: int = None):

    data = data.copy()

    mask = data['PredictedPrice'] == -np.inf
    data.loc[mask, 'PredictedPrice'] = None

    if not isinstance(data.index, pd.DatetimeIndex):
        print("❌ 'data' must have a datetime index.")
        return

    data.rename(columns={
        'ActualPrice': 'Actual',
        'PredictedPrice': 'Predicted'
    }, inplace=True)

    if n is not None:
        test_df = data[data['Predicted'].notna()].copy()
        last_n = test_df.tail(n)
        min_dt = last_n.index.min().normalize()
        data = test_df[test_df.index >= min_dt].copy()

    data['x_str'] = data.index.strftime('%Y-%m-%d %H:%M')
    seen_days = set()
    x_tickvals = []
    x_ticktext = []
    for i, ts in enumerate(data.index):
        day_label = ts.strftime('%Y-%m-%d')
        if day_label not in seen_days:
            seen_days.add(day_label)
            x_tickvals.append(data['x_str'].iloc[i])
            x_ticktext.append(ts.strftime('%b %d'))

    y_min = math.floor(min(data['Actual'].min(), data['Predicted'].min()))
    y_max = math.ceil(max(data['Actual'].max(), data['Predicted'].max()))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data['x_str'],
        y=data['Actual'],
        mode='lines+markers',
        line=dict(color='blue', width=2),
        marker=dict(size=4),
        name='Actual',
        hovertemplate='Actual: %{x}<br>₹%{y:,.2f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=data['x_str'],
        y=data['Predicted'],
        mode='lines+markers',
        line=dict(color='red', width=2, dash='dot'),
        marker=dict(size=4),
        name='Predicted',
        hovertemplate='Predicted: %{x}<br>₹%{y:,.2f}<extra></extra>'
    ))

    fig.update_layout(
        title=f'{ticker} - Prediction{" (last ~" + str(n) + " points)" if n else ""}',
        template='plotly_dark',
        xaxis=dict(
            title='Time',
            type='category',
            tickmode='array',
            tickvals=x_tickvals,
            ticktext=x_ticktext,
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            range=[y_min-500, y_max+500],
            title='Price (₹)',
            tickprefix='₹',
            tickformat=',.0f',
            showgrid=False
        ),
        hovermode='x unified'
    )

    fig.show()


# ------------------------------------------------------------------

def interactive_data_plot(data: pd.DataFrame, ticker: str, n: int = None):
    """
    Plots an interactive line chart of actual stock prices.

    Args:
        data (pd.DataFrame): DataFrame containing stock data with a datetime index.
        ticker (str): The stock ticker symbol.
        n (int, optional): The number of recent data points to plot.
                                  If None, the entire dataset is plotted.
    """
    data = data.copy()

    if not isinstance(data.index, pd.DatetimeIndex):
        print("❌ 'data' must have a datetime index.")
        return

    data.rename(columns={
        'ActualPrice': 'Actual'
    }, inplace=True)

    if n is not None:
        last_n = data.tail(n)
        min_dt = last_n.index.min().normalize()
        data =  data[data.index >= min_dt].copy()
    
    # Data preprocessing for plotting
    data['x_str'] = data.index.strftime('%Y-%m-%d %H:%M')
    seen_days = set()
    x_tickvals = []
    x_ticktext = []
    for i, ts in enumerate(data.index):
        day_label = ts.strftime('%Y-%m-%d')
        if day_label not in seen_days:
            seen_days.add(day_label)
            x_tickvals.append(data['x_str'].iloc[i])
            x_ticktext.append(ts.strftime('%b %d'))

    y_min = math.floor(data['Actual'].min())
    y_max = math.ceil(data['Actual'].max())

    fig = go.Figure()

    # The only plotting trace, focusing on 'Actual' prices.
    fig.add_trace(go.Scatter(
        x=data['x_str'],
        y=data['Actual'],
        mode='lines+markers',
        line=dict(color='blue', width=2),
        marker=dict(size=4),
        name='Actual Price',
        hovertemplate='Actual: %{x}<br>₹%{y:,.2f}<extra></extra>'
    ))

    fig.update_layout(
        title=f'Actual Price for {ticker}{" (last ~" + str(n) + " points)" if n else ""}',
        template='plotly_dark',
        xaxis=dict(
            title='Time',
            type='category',
            tickmode='array',
            tickvals=x_tickvals,
            ticktext=x_ticktext,
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            range=[y_min - 500, y_max + 500],
            title='Price (₹)',
            tickprefix='₹',
            tickformat=',.0f',
            showgrid=False
        ),
        hovermode='x unified'
    )

    fig.show()

# #------------------------------------------------------------------

# def interactive_data_plot(filepath: str, ticker: str):
#     if not os.path.exists(filepath):
#         print(f"❌ Error: File '{filepath}' not found.")
#         return

#     try:
#         df = pd.read_csv(filepath, index_col=0)
#         df.index = pd.to_datetime(df.index, errors='coerce')
#         df = df[~df.index.isna()]  # remove bad timestamps
#         df.columns = df.columns.str.strip()

#         if 'ActualPrice' not in df.columns:
#             print("❌ Error: 'ActualPrice' column not found.")
#             return

#         df['ActualPrice'] = pd.to_numeric(df['ActualPrice'], errors='coerce')
#         df.dropna(subset=['ActualPrice'], inplace=True)

#         # Last 100 data points + full day context
#         last_100_df = df.tail(100).copy()
#         first_date = last_100_df.index[0].date()
#         start_index = df[df.index.date < first_date].shape[0]
#         df_tail = df.iloc[start_index:].copy()

#         # Add DateOnly and formatted time
#         df_tail['DateOnly'] = df_tail.index.date
#         df_tail['TimeLabel'] = df_tail.index.strftime('%b %d %H:%M')

#         # X: stringified timestamps
#         x_labels = df_tail.index.strftime('%b %d %H:%M')  # for plotting
#         x_tickvals = []
#         x_ticktext = []

#         # Show only one label per day on x-axis
#         seen_days = set()
#         for i, ts in enumerate(df_tail.index):
#             day_label = ts.strftime('%b %d')
#             if day_label not in seen_days:
#                 seen_days.add(day_label)
#                 x_tickvals.append(x_labels[i])
#                 x_ticktext.append(day_label)

#         # Y-axis range
#         min_ActualPrice = df_tail['ActualPrice'].min()
#         max_ActualPrice = df_tail['ActualPrice'].max()
#         y_min = math.floor(min_ActualPrice - 2)
#         y_max = math.ceil(max_ActualPrice + 2)

#         fig = go.Figure()

#         fig.add_trace(go.Scatter(
#             x=x_labels,  # string-based timestamps (categorical)
#             y=df_tail['ActualPrice'],
#             mode='lines+markers',
#             line=dict(color='cyan', width=2),
#             marker=dict(size=5, color='orange'),
#             name='ActualPrice Price',
#             hovertemplate='Timestamp: %{x}<br>Price: %{y:,.2f}<extra></extra>'
#         ))

#         fig.update_layout(
#             title=f'ActualPrice Price for {ticker} (Last 100+ Points)',
#             template='plotly_dark',
#             xaxis=dict(
#                 title='Date',
#                 type='category',
#                 tickmode='array',
#                 tickvals=x_tickvals,
#                 ticktext=x_ticktext,
#                 tickangle=0,
#                 tickfont=dict(size=10),
#                 showgrid=True,
#                 gridcolor='rgba(255,255,255,0.1)',
#             ),
#             yaxis=dict(
#                 range=[y_min, y_max],
#                 title='ActualPrice Price',
#                 tickprefix='₹',
#                 tickformat=',.0f',
#                 showgrid=False
#             ),
#             hovermode='x unified'
#         )

#         fig.show()

#     except Exception as e:
#         print(f"❌ An error occurred while plotting: {e}")

# ------------------------------------------------------------------
import pandas as pd
import plotly.graph_objects as go
import math
import numpy as np
import os
def plot_train_and_prediction_and_last_pred_point(data: pd.DataFrame, ticker: str, n: int = None, last_pred=None):

    data = data.copy()

    mask = data['PredictedPrice'] == -np.inf
    data.loc[mask, 'PredictedPrice'] = None

    if not isinstance(data.index, pd.DatetimeIndex):
        print("❌ 'data' must have a datetime index.")
        return

    data.rename(columns={
        'ActualPrice': 'Actual',
        'PredictedPrice': 'Predicted'
    }, inplace=True)

    if n is not None:
        test_df = data[data['Predicted'].notna()].copy()
        last_n = test_df.tail(n)
        min_dt = last_n.index.min().normalize()
        data = test_df[test_df.index >= min_dt].copy()

    data['x_str'] = data.index.strftime('%Y-%m-%d %H:%M')
    seen_days = set()
    x_tickvals = []
    x_ticktext = []
    for i, ts in enumerate(data.index):
        day_label = ts.strftime('%Y-%m-%d')
        if day_label not in seen_days:
            seen_days.add(day_label)
            x_tickvals.append(data['x_str'].iloc[i])
            x_ticktext.append(ts.strftime('%b %d'))

    y_min = math.floor(min(data['Actual'].min(), data['Predicted'].min()))
    y_max = math.ceil(max(data['Actual'].max(), data['Predicted'].max()))

    fig = go.Figure()

    # Actual
    fig.add_trace(go.Scatter(
        x=data['x_str'],
        y=data['Actual'],
        mode='lines+markers',
        line=dict(color='blue', width=2),
        marker=dict(size=4),
        name='Actual',
        hovertemplate='Actual: %{x}<br>₹%{y:,.2f}<extra></extra>'
    ))

    # Predicted
    fig.add_trace(go.Scatter(
        x=data['x_str'],
        y=data['Predicted'],
        mode='lines+markers',
        line=dict(color='red', width=2, dash='dot'),
        marker=dict(size=4),
        name='Predicted',
        hovertemplate='Predicted: %{x}<br>₹%{y:,.2f}<extra></extra>'
    ))

    # Last prediction (next hour price)
    if last_pred is not None:
        last_pred_datetime = data.index[-1] + pd.Timedelta(hours=1)
        last_pred_x = last_pred_datetime.strftime('%Y-%m-%d %H:%M')
        fig.add_trace(go.Scatter(
            x=[data['x_str'].iloc[-1], last_pred_x],
            y=[data['Predicted'].iloc[-1], last_pred],
            mode='lines+markers',
            line=dict(color='yellow', width=2, dash='dot'),
            marker=dict(size=6, color='yellow'),
            name='Next Hour Price',
            hovertemplate='Next hour price: %{x}<br>₹%{y:,.2f}<extra></extra>'
        ))

    fig.update_layout(
        title=f'{ticker} - Prediction{" (last ~" + str(n) + " points)" if n else ""}',
        template='plotly_dark',
        xaxis=dict(
            title='Time',
            type='category',
            tickmode='array',
            tickvals=x_tickvals,
            ticktext=x_ticktext,
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            range=[y_min-500, y_max+500],
            title='Price (₹)',
            tickprefix='₹',
            tickformat=',.0f',
            showgrid=False
        ),
        hovermode='x unified'
    )

    fig.show()

