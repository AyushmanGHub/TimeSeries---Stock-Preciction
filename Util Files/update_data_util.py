import yfinance as yf
import pandas as pd
import numpy as np
import os
import warnings

# Suppress specific pandas warnings
warnings.filterwarnings('ignore', message='Could not infer format')

def fetch_and_update_stock_data(ticker_symbol: str, data_dir: str = '.', timezone='Asia/Kolkata'):
    """
    Fetches 2 years of hourly stock data for a given ticker and updates a local CSV file
    by appending new data. It also explicitly cleans up the header format on creation
    and only if needed on update.

    Args:
        ticker_symbol (str): The ticker symbol of the stock (e.g., '^NSEBANK').
        data_dir (str, optional): The directory where the data will be saved. Defaults to the current directory.
        timezone (str, optional): The timezone for the data. Defaults to 'Asia/Kolkata'.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        ticker_obj = yf.Ticker(ticker_symbol)
        company_name = ticker_obj.info.get("shortName", ticker_symbol).replace(" ", "_").upper()
    except Exception:
        company_name = ticker_symbol.replace("^", "").replace(".", "_").upper()
    
    filename = f"{company_name}_data.csv"
    filepath = os.path.join(data_dir, filename)
    period = '2y'
    
    def cleanup_and_insert_header(file_path):
        """Deletes the first 3 lines and inserts a clean header if needed."""
        if not os.path.exists(file_path):
            return
            
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            clean_header = "Datetime,Price,Close,High,Low,Open,Volume\n"
            
            # Check if the file already has the clean header to avoid redundant writes
            if lines and lines[0] == clean_header:
                return

            # Keep all lines from the 4th line onwards (index 3)
            data_lines = lines[3:]
            
            # Overwrite the file with the new header and the old data
            with open(file_path, 'w') as f:
                f.write(clean_header)
                f.writelines(data_lines)
            
            print(f"🧹 Header cleaned and new header inserted in '{os.path.basename(file_path)}'.")

        except Exception as e:
            print(f"❌ Error during header cleanup of '{os.path.basename(file_path)}': {e}")

    # --- MAIN LOGIC ---
    if os.path.exists(filepath):
        print(f"📈 Updating existing data for {company_name}...")
        
        # Conditionally clean up the header only if it's malformed
        cleanup_and_insert_header(filepath)

        try:
            existing_data = pd.read_csv(
                filepath,
                index_col='Datetime',
                parse_dates=True
            )
            
            existing_data.index = existing_data.index.tz_convert(timezone)
            last_date_in_file = existing_data.index[-1]
            print(f"Last record in file is from: {last_date_in_file}")

            new_data_full = yf.download(
                ticker_symbol,
                start=last_date_in_file,
                interval="1h",
                progress=False,
                auto_adjust=True
            )
            
            if not new_data_full.empty:
                new_data_full.index = new_data_full.index.tz_convert(timezone)
                
                new_data_to_append = new_data_full[new_data_full.index > last_date_in_file]
                
                if not new_data_to_append.empty:
                    if 'Adj Close' in new_data_to_append.columns:
                        new_data_to_append = new_data_to_append.drop('Adj Close', axis=1)
                    
                    new_data_to_append.insert(0, 'Price', new_data_to_append['Close'])
                    target_columns = ['Price', 'Close', 'High', 'Low', 'Open', 'Volume']
                    new_data_to_append = new_data_to_append[target_columns]
                    
                    # Print the update range
                    first_new_record_date = new_data_to_append.index[0]
                    last_new_record_date = new_data_to_append.index[-1]
                    print(f"✅ Data updated from {first_new_record_date} to {last_new_record_date}")
                    
                    new_data_to_append.to_csv(filepath, mode='a', header=False)
                    print(f"📊 Added {len(new_data_to_append)} new records")
                    return new_data_to_append
                else:
                    print("⚠️ No new data found after the last record in the file.")
                    return "No data point to update"
            else:
                print("⚠️ No new data found to update.")
                return "No data point to update"
        
        except Exception as e:
            print(f"Error during update: {e}")
            print("Attempting a fresh download...")
            os.remove(filepath)
            return fetch_and_update_stock_data(ticker_symbol, data_dir, timezone)
    
    else:
        print(f"📦 No existing data found. Fetching full {period} history for {company_name}...")
        
        stock_data = yf.download(
            ticker_symbol,
            period=period,
            interval="1h",
            progress=False,
            auto_adjust=True
        )
        
        if not stock_data.empty:
            stock_data.index = stock_data.index.tz_convert(timezone)
            
            if 'Adj Close' in stock_data.columns:
                stock_data = stock_data.drop('Adj Close', axis=1)
            
            stock_data.insert(0, 'Price', stock_data['Close'])
            stock_data.index.name = 'Datetime'
            
            stock_data.to_csv(filepath, index=True, header=True)
            cleanup_and_insert_header(filepath) # Cleans up immediately on creation

            print(f"✅ Data downloaded and saved to: '{filepath}'")
            print(f"📊 Saved {len(stock_data)} records")
            return stock_data
        else:
            print(f"❌ No data found for ticker: {ticker_symbol}")
            return "No data point to update"
        


def update_predictions_file(data_file_path, predictions_file_path, n_rows=300, val_to_replace=-np.inf):
    """
    Updates or creates a predictions CSV file based on new data.

    If the predictions file exists, it appends any new data points
    from the data file. Otherwise, it creates a new predictions file
    containing all data points from the data file.

    Args:
        data_file_path (str): The file path to the source data CSV.
        predictions_file_path (str): The file path to the predictions CSV.
    """
    n_rows = n_rows - 1

    # Read the data file and make the 'Datetime' column time zone-naive
    data_df = pd.read_csv(data_file_path, parse_dates=['Datetime'])
    data_df['Datetime'] = data_df['Datetime'].dt.tz_localize(None)

    if os.path.exists(predictions_file_path):
        try:
            # Read the existing predictions file and parse 'Datetime'
            predictions_df = pd.read_csv(predictions_file_path, parse_dates=['Datetime'], on_bad_lines="skip")
            
            if not predictions_df.empty:
                # Make the 'Datetime' column of predictions_df time zone-naive
                predictions_df['Datetime'] = predictions_df['Datetime'].dt.tz_localize(None)
                last_datetime = predictions_df['Datetime'].iloc[-1]
                
                # Filter for new entries
                new_data = data_df[data_df['Datetime'] > last_datetime]
                
                if not new_data.empty:
                    # Prepare the new data to be appended
                    new_data = new_data[["Datetime", "Close"]].copy()
                    new_data.rename(columns={'Close': 'ActualPrice'}, inplace=True)
                    new_data['PredictedPrice'] = None
                    
                    # Append to the predictions file without writing the header
                    new_data.to_csv(predictions_file_path, mode='a', index=False, header=False)
                    print(f"Appended {len(new_data)} new entries to {predictions_file_path}")
                else:
                    print(f"No new data to append to {predictions_file_path}. The file is up to date.")
            else:
                # If predictions file exists but is empty, create it from scratch
                print("Predictions file is empty. Creating it with all available data.")
                data_df = data_df[["Datetime", "Close"]].copy()
                data_df.rename(columns={'Close': 'ActualPrice'}, inplace=True)
                data_df['PredictedPrice'] = None
                data_df.iloc[:n_rows, data_df.columns.get_loc('PredictedPrice')] = val_to_replace
                data_df.to_csv(predictions_file_path, index=False)
                print(f"Created a new file at {predictions_file_path} with all data.")
        
        except pd.errors.EmptyDataError:
            print("Predictions file is empty. Creating it with all available data.")
            # Handle the case where the file exists but is empty
            data_df_processed = data_df[["Datetime", "Close"]].copy()
            data_df_processed.rename(columns={'Close': 'ActualPrice'}, inplace=True)
            data_df_processed['PredictedPrice'] = None
            data_df_processed.iloc[:n_rows, data_df_processed.columns.get_loc('PredictedPrice')] = val_to_replace
            data_df_processed.to_csv(predictions_file_path, index=False)
            print(f"Created a new file at {predictions_file_path} with all data.")
    else:
        # If the predictions file does not exist, create it from the source data
        data_df_processed = data_df[["Datetime", "Close"]].copy()
        data_df_processed.rename(columns={'Close': 'ActualPrice'}, inplace=True)
        data_df_processed['PredictedPrice'] = None
        data_df_processed.iloc[:n_rows, data_df_processed.columns.get_loc('PredictedPrice')] = val_to_replace
        data_df_processed.to_csv(predictions_file_path, index=False)
        print(f"Created a new file at {predictions_file_path} with all data.")