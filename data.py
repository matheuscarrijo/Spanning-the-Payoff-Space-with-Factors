import pandas as pd
import pandas_datareader as pdr
import warnings # Ignore warnings
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def crsp_m(data_path, start_date="1973-01-31", end_date="2023-12-31",
           missing_data_info=False):
    """
    Load and process the CRSP Monthly dataset in chunks, applying filters,
    adjustments for delisting returns, and macro integration.
    
    Parameters:
    - data_path (str): Path to the CRSP dataset (.csv).
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.
    - missing_data_info (bool): If True, return a DataFrame with missing data 
      summary.
    
    Returns:
    - pd.DataFrame: Cleaned CRSP monthly data.
    - pd.DataFrame (optional): Missing data summary if missing_data_info=True.
    """
    
    # Define the file path 
    #data_path = "data/CRSP-Stock-Monthly.csv"  
    
    # Set dates 
    #start_date = pd.to_datetime("1926-01-31") # Entire CRSP sample
    #start_date = "1962-07-31" # AMEX birth. 
    
    start_date = pd.to_datetime(start_date) # NASDAQ birth is 1972-12-31
    end_date = pd.to_datetime(end_date)
    
    # Guard against start_date earlier than Fama-French data availability
    MIN_START_DATE = pd.to_datetime("1926-07-01")
    if start_date < MIN_START_DATE:
        raise ValueError(f"start_date must be on or after {MIN_START_DATE.date()} to match Fama-French data availability.")

    #########################
    # CPI data from FRED
    #########################
    
    cpi_m = (pdr.DataReader(name="CPIAUCNS", data_source="fred", 
                            start="1972-12-31", end=end_date)
                .reset_index(names="month")
                .rename(columns={"CPIAUCNS": "cpi"})
                .assign(cpi=lambda x: x["cpi"]/x["cpi"].iloc[-1]))
    
    cpi_m['month'] = (cpi_m['month'].dt.to_period('M').dt.to_timestamp('M'))

    ##################################################
    # Risk-free rate (RF) from Fama-French database
    ##################################################
    
    rf_m  = (pdr.DataReader(name="F-F_Research_Data_Factors", 
                            data_source="famafrench", start=start_date, 
                            end=end_date)[0][['RF']].divide(100)
                            .reset_index(names="month")
                            .assign(month=lambda x: 
                                    pd.to_datetime(x["month"].astype(str)))
                            .rename(str.lower, axis="columns"))
    rf_m['month'] = (rf_m['month'].dt.to_period('M').dt.to_timestamp('M'))

    #########################
    # CRSP stock returns:
    #########################
    
    # List of columns to keep 
    columns_to_keep = [
        'date', 'PERMNO', 'SHRCD', 'RET', 'DLRET', 'DLSTCD', 'SHROUT', 'PRC',
        'EXCHCD', 'COMNAM', 'SICCD'
    ]
    
    # Define the chunk size
    chunk_size = 100000
    
    # Initialize an empty list to store results
    data_list = []
    
    # Iterate through the file in chunks and read only the necessary columns
    for chunk in pd.read_csv(data_path, usecols=columns_to_keep, 
                             chunksize=chunk_size):
        
        # Convert columns to numeric, forcing non-numeric values to NaN
        chunk['RET'] = pd.to_numeric(chunk['RET'], errors='coerce')
        chunk['DLRET'] = pd.to_numeric(chunk['DLRET'], errors='coerce')
        chunk['SICCD'] = pd.to_numeric(chunk['SICCD'], errors='coerce')
        
        # CRSP originally provides the share outstanding in thousand. We "fix" 
        # that.
        chunk['SHROUT'] = chunk['SHROUT'] * 1000
        
        # Convert date column to datetime format
        chunk['date'] = pd.to_datetime(chunk['date'], format='%Y-%m-%d')
    
        # Rename columns to lowercase and replace 'date' by 'month'
        chunk.columns = [col.lower() for col in chunk.columns]
        chunk = chunk.rename(columns={'date': 'month'})
    
        # Adjust all dates to the end of the month
        chunk['month'] = chunk['month'].dt.to_period('M').dt.to_timestamp('M')

        # FILTERS:
    
        # Filter the data to load only records starting from 'start_date_1973'
        chunk = chunk[chunk['month'] >= start_date]
    
        # Apply filters for specific exchanges (NYSE, AMEX, NASDAQ)
        chunk_filtered = chunk[chunk["exchcd"].isin([1, 31, 2, 32, 3, 33])].copy()
        
        # Apply filters for ordinary common stocks (share codes 10 and 11)
        chunk_filtered = chunk_filtered[chunk_filtered["shrcd"].isin([10, 11])].copy()
    
        # Exclude financial firms (SIC codes between 6000 and 6999)
        chunk_filtered = chunk_filtered[~chunk_filtered['siccd'].between(6000, 6999)].copy()
        
        # Adjust dlret for specific delisting codes and exchanges as in 
        # Shumway (1997) and Shumway and Warther (1999)
        chunk_filtered['dlret'] = chunk_filtered.apply(
            lambda row: -0.30 if (
                pd.isna(row['dlret']) and row['dlstcd'] in range(500, 585) and 
                row['exchcd'] in [1, 31, 2, 32]
            ) else (
                -0.55 if (
                    pd.isna(row['dlret']) and row['dlstcd'] in range(500, 585) and
                    row['exchcd'] in [3, 33]
                ) else row['dlret']
            ),
            axis=1
        )
    
        # Cap dlret to a minimum of -1
        chunk_filtered['dlret'] = chunk_filtered['dlret'].apply(
            lambda x: max(x, -1) if not pd.isna(x) else x
        )
    
        # Fill missing delisting returns with 0
        chunk_filtered['dlret'] = chunk_filtered['dlret'].fillna(0)
    
        # Adjust the returns using the formula: (1 + ret) * (1 + dlret) - 1
        chunk_filtered['ret'] = (
            (1 + chunk_filtered['ret']) * (1 + chunk_filtered['dlret']) - 1
        )
    
        # If ret is missing but dlret is not, set ret = dlret
        chunk_filtered['ret'] = chunk_filtered.apply(
            lambda row: row['dlret'] if pd.isna(row['ret']) and row['dlret'] != 0 
            else row['ret'], axis=1
        )
        
        # Append the filtered chunk to the list
        data_list.append(chunk_filtered)
        
    # Concatenate all filtered chunks into a single DataFrame
    crsp_m = pd.concat(data_list, axis=0).reset_index(drop=True)
    # Ensure 'month' is in datetime format for accurate sorting
    crsp_m['month'] = pd.to_datetime(crsp_m['month'])

    # Filter out firms that have less than 60 observations as in one of the
    # trimming conditions discussed in Gagliardini, Ossola and Scaillet (2019).
    obs_per_permno = crsp_m.groupby('permno').size().reset_index(name='n_obs')
    valid_permnos = obs_per_permno[obs_per_permno['n_obs'] >= 60]['permno']
    crsp_m = crsp_m[crsp_m['permno'].isin(valid_permnos)]
    
    # Select only stocks where the minimum price ('prc') across all observations 
    # is greater than $5
    permnos_with_high_price = crsp_m.groupby('permno')['prc'].min()
    selected_permnos = permnos_with_high_price[permnos_with_high_price > 5].index
    # Filter the original dataset to include only these selected stocks
    crsp_m = crsp_m[crsp_m['permno'].isin(selected_permnos)]
    
    # Integrate the risk-free rate from Fama-French data into the CRSP monthly 
    # data to calculate excess returns. The excess return is the return earned 
    # by a security or portfolio over the risk-free rate.
    crsp_m = (crsp_m
      .merge(rf_m[["month", "rf"]], how="left", on="month")
      .assign(ret_excess=lambda x: x["ret"]-x["rf"])
      .assign(ret_excess=lambda x: x["ret_excess"].clip(lower=-1))
      .drop(columns=["rf"])
    )
    
    # Adding market cap
    crsp_m = (crsp_m
      .assign(mktcap=lambda x: abs(x["shrout"]*x["prc"]))) 
    crsp_m['mktcap'] = crsp_m['mktcap'].round(2)
    
    # Transform the primary listing exchange codes in the CRSP dataset into 
    # explicit exchange names
    def assign_exchange(exchcd):
        if (exchcd == 1) or (exchcd == 31):
            return "NYSE"
        elif (exchcd == 2) or (exchcd == 32):
            return "AMEX"
        elif (exchcd == 3) or (exchcd == 33):
            return "NASDAQ"
        else: 
            return "Other"
    
    crsp_m["exchange"] = crsp_m["exchcd"].apply(assign_exchange)
    
    # Transform SIC codes in the CRSP dataset into explicit industry sector names
    def assign_industry(siccd):
        if 100 <= siccd <= 999:
            return "Agriculture"
        elif 1000 <= siccd <= 1499:
            return "Mining"
        elif 1500 <= siccd <= 1799:
            return "Construction"
        elif 2000 <= siccd <= 3999:
            return "Manufacturing"
        elif 4000 <= siccd <= 4999:
            return "Utilities"
        elif 5000 <= siccd <= 5199:
            return "Wholesale"
        elif 5200 <= siccd <= 5999:
            return "Retail"
        elif 6000 <= siccd <= 6999:
            return "Finance"
        elif 7000 <= siccd <= 8999:
            return "Services"
        elif 9100 <= siccd <= 9729:
            return "Public"
        elif 9900 <= siccd <= 9999:
            return "Nonclassifiable"
        else:
            return "Missing"
    
    crsp_m["industry"] = crsp_m["siccd"].apply(assign_industry)
    
    
    if missing_data_info is True: 
        
        # INSPECTING MISSING DATA IN `crsp_m` DATASET
        # Group by 'permno' and count the missing values in 'RET'
        missing_returns_count = (crsp_m.groupby('permno')['ret_excess'].apply(lambda x: 
                                 x.isna().sum()))
            
        # Calculate the percentage of missing values for each stock
        total_observations_per_stock = crsp_m.groupby('permno')['ret_excess'].size()
        missing_returns_percentage = ((missing_returns_count / 
                                       total_observations_per_stock) * 100)
        
        # Create a DataFrame to summarize the missing values count and percentage
        missing_data_summary = pd.DataFrame({
            'Missing Count': missing_returns_count,
            'Total Observations': total_observations_per_stock,
            'Missing Percentage': missing_returns_percentage
        })
        
        # Sort by the number or percentage of missing values
        missing_data_summary_sorted = (missing_data_summary
                                       .sort_values(by='Missing Percentage', 
                                        ascending=False))
        
        # Replace missing excess returns with 0
        crsp_m['ret_excess'] = (crsp_m['ret_excess'].fillna(0))

        return crsp_m, missing_data_summary_sorted
    
    # Replace missing excess returns with 0
    crsp_m['ret_excess'] = (crsp_m['ret_excess'].fillna(0))
    
    return crsp_m

def gfd(gfd_path, start_date="1973-01-31", end_date="2023-12-31"):
    """
    Load Global Factor Data (GFD) plus market factor from Fama-French dataset.

    Parameters:
    - gfd_path (str): Path to the Global Factor Data.
    - start_date (str): Start date for filtering Global Factor Data.
    - end_date (str): End date for Fama-French data.

    Returns:
    - pd.DataFrame: Observed factors panel with GFD factors + MKT-RF.
    """
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Guard against start_date before July 1926 (Fama-French data availability)
    MIN_START_DATE = pd.to_datetime("1926-07-01")
    if start_date < MIN_START_DATE + pd.DateOffset(months=1):
        raise ValueError(f"start_date must be at least one month after {MIN_START_DATE.date()} to retrieve Fama-French data.")
    
    # Calculate one-month-before date for FF data
    ff_start_date = start_date - pd.DateOffset(months=1)

    # Load MKT-RF from Fama-French
    mkt_factor = (pdr.DataReader("F-F_Research_Data_Factors", "famafrench",
                                 start=ff_start_date, 
                                 end=end_date)[0][['Mkt-RF']]
                    .divide(100)
                    .reset_index(names="month")
                    .assign(month=lambda x: pd.to_datetime(x["month"].astype(str)))
                    .rename(str.lower, axis="columns"))
    mkt_factor['month'] = (mkt_factor['month'].dt.to_period('M')
                                              .dt.to_timestamp('M'))

    # Load GFD data
    obs_F = pd.read_csv(gfd_path)
    obs_F['date'] = pd.to_datetime(obs_F['date'])
    obs_F = obs_F[obs_F['date'] >= pd.to_datetime("1973-03-31")]
    obs_F = obs_F.pivot_table(index='date', columns='name', values='ret')

    # Merge MKT-RF with GFD factors
    mkt_factor['month'] = pd.to_datetime(mkt_factor['month'])
    obs_F = obs_F.reset_index()
    obs_F['date'] = pd.to_datetime(obs_F['date'])
    merged_df = obs_F.merge(mkt_factor[['month', 'mkt-rf']], 
                            left_on='date', right_on='month', how='left')

    # Reorganize and format
    columns = (['date', 'mkt-rf'] + 
               [col for col in obs_F.columns if 
                col not in ['date', 'mkt-rf']])
    obs_F = merged_df[columns].set_index('date')
    obs_F.index = pd.to_datetime(obs_F.index)

    return obs_F

def return_panels(crsp_df):
    """
    Build return panels (permno-based, company-name based, and lagged) 
    from CRSP monthly data.

    Parameters:
    - crsp_df (pd.DataFrame): Output from `crsp_m(...)`, must contain
      'month', 'permno', 'comnam', 'ret_excess'.

    Returns:
    - ret_panel (pd.DataFrame): Return panel with permno columns.
    - ret_panel_comnam (pd.DataFrame): Return panel with company names.
    - aug_ret_panel (pd.DataFrame): Augmented panel with lagged returns.
    """
    
    # Pivot by permno
    ret_panel = crsp_df.pivot_table(index='month', columns='permno', values='ret_excess')
    ret_panel.index = ret_panel.index.strftime('%Y-%m-%d')

    # Insert "Transform:" row with 1s
    transform_row = pd.DataFrame([[1]*len(ret_panel.columns)], 
                                 columns=ret_panel.columns, index=['Transform:'])
    ret_panel = pd.concat([transform_row, ret_panel])

    # Clean version without transformation row
    ret_pivot = ret_panel.drop(index="Transform:", errors="ignore")

    # Replace permno with company names
    crsp_df['month'] = pd.to_datetime(crsp_df['month'])
    crsp_sorted = crsp_df.sort_values(by=['permno', 'month'], ascending=[True, False])
    permno_to_comnam = (crsp_sorted
                        .drop_duplicates(subset='permno', keep='first')[['permno', 'comnam']]
                        .set_index('permno')['comnam'])
    ret_panel_comnam = ret_pivot.rename(columns=permno_to_comnam)

    # Augmented panel with lags
    lagged_returns = ret_pivot.shift(1)
    lagged_returns.columns = [f"{col}_lag1" for col in lagged_returns.columns]
    aug_ret_pivot = pd.concat([ret_pivot, lagged_returns], axis=1).iloc[1:]

    transform_row = pd.DataFrame([[1]*len(aug_ret_pivot.columns)], 
                                 columns=aug_ret_pivot.columns, index=['Transform:'])
    aug_ret_panel = pd.concat([transform_row, aug_ret_pivot])

    return ret_panel, ret_panel_comnam, aug_ret_panel
