import numpy as np
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime
    
def data_preprocessing():

    """
     This function process the datasets used (explicitly or implicitly) in 
    the paper. The preprocessing steps for each dataset is explained below.

    ---------------------------------------------------------------------------
    
                                  CPI data from FRED
                                  
    Below we retrieves the Consumer Price Index (CPI) data from the Federal 
    Reserve Economic Data (FRED) system for the period specified by the dates.
    
    - The dataset "CPIAUCNS" refers to the Consumer Price Index for All Urban 
      Consumers, not Seasonally Adjusted, which measures inflation by tracking 
      price changes of goods and services consumed by urban households.
    - `pdr.DataReader()` fetches this data from the "fred" source.
    - The date index is reset to a standard column named "month" to facilitate
      merging with other datasets.
    - The column is renamed from "CPIAUCNS" to "cpi" for clarity.
    - The CPI values are normalized by dividing them by the last available CPI 
      value, making the series relative to the most recent month.
    - The "month" column is converted to represent the last day of each month 
      to ensure consistency in time-based analyses.

    ---------------------------------------------------------------------------
    
                          Risk-free rate (RF) from FF database
                          
    Below we retrieves the risk-free rate (RF) data from the Fama-French 
    database for the period between 1972-12-31 and the predefined end date. 
    
    - The data is extracted using `pdr.DataReader()` from the "famafrench" 
      source, 
      specifically from the "F-F_Research_Data_Factors" dataset.
    - The first element of the returned dictionary (`[0]`) contains monthly 
      factor data, 
      from which only the "RF" (risk-free rate) column is selected.
    - The RF values are originally in percentage format, so they are divided by
      100 to 
      convert them to decimal form.
    - The date index is reset to a standard column called "month".
    - The "month" column is explicitly converted to datetime format to avoid 
      inconsistencies.
    - Column names are converted to lowercase for uniformity.
    - Finally, the "month" column is converted to represent the last day of 
      each month to maintain consistency with other financial time series 
      datasets.
      
    ---------------------------------------------------------------------------

                               CRSP stock returns:
                               
    Below the code processes the CRSP monthly stock data by loading, filtering,
    and cleaning it in chunks to handle large datasets efficiently. 
    
    Data Loading and Cleaning
       - Reads the CRSP monthly stock dataset in chunks to optimize memory usage.
       - Keeps only relevant columns and ensures numeric conversion where 
         necessary.
       - Converts the date column to datetime format and adjusts all dates to 
         end-of-month.
    
    Data Filtering
       - Filters stocks listed on NYSE, AMEX, and NASDAQ.
       - Selects only common stocks (SHRCD 10 and 11).
       - Excludes financial firms (SIC codes 6000-6999).
       - Adjusts delisting returns using methodologies from Shumway (1997) and 
         Shumway & Warther (1999).
       - Removes stocks with fewer than 60 monthly observations.
       - Retains only stocks with a minimum historical price above $5.
    
    Return Adjustments
       - Adjusts returns based on delisting returns.
       - Computes excess returns by subtracting the risk-free rate from raw 
         returns.
    
    Market Capitalization Calculation
       - Computes market capitalization as `SHROUT * PRC` and rounds to two 
         decimals.
    
    Categorical Variable Transformations
       - Maps exchange codes to readable names (NYSE, AMEX, NASDAQ).
       - Converts SIC codes into industry sector classifications.
          
    ---------------------------------------------------------------------------
    
                                  Observed factors
        
       This script part processes the Market Factor (MKT-RF) from the
       Fama-French library and integrates it with observed factors from the
       Global Factor Data (GFD).
    
    Fama-French Market Factor (MKT-RF)
       - The Fama-French market factor is retrieved from the
         "F-F_Research_Data_Factors" dataset.
       - The data is extracted starting from December 31, 1972.
       - The factor is converted from percentage format to decimal.
       - The date column is reformatted and set to the last day of each month.
    
    Observed Factors from Global Factor Data (GFD)
       - The dataset is loaded from a CSV file containing multiple factors.
       - The 'date' column is converted to datetime format.
       - Data is filtered to start from March 31, 1973.
       - A pivot table is created, restructuring the data to have factors as
         columns.
    
    Merging Fama-French and GFD Factors
       - The 'month' column in the market factor dataset is converted to
         datetime.
       - The observed factors dataset is temporarily reset to ensure proper
         merging.
       - The two datasets are merged on the corresponding date columns using a
         left join.
       - The final dataset is reorganized, ensuring 'mkt-rf' appears
         immediately after the date.
       - The 'date' column is restored as the index, ensuring it remains a
         properly formatted datetime index.
    
    The resulting dataset combines the MKT-RF factor from Fama-French with
    various observed factors from GFD.
    
    ---------------------------------------------------------------------------
    
                                ret_panel
    
    This dataset constructs a panel of excess returns by organizing CRSP stock
    return data into a structured time series format.
    
    - The dataset is structured as a pivot table with stock identifiers (permno)
      as columns and months as index values.
    - The data represents excess returns (returns adjusted for risk-free rate).
    - The resulting dataset facilitates analysis of return time series across
      multiple stocks.
    - A transformation row is added to indicate that all variables are directly
      used in the analysis.
    
    ---------------------------------------------------------------------------
    
                                ret_panel_comnam
    
    This dataset is an alternative representation of `ret_panel`, where columns
    are labeled using company names instead of stock identifiers (permno).
    
    - The most recent company name associated with each permno is extracted.
    - The column names of `ret_panel` are replaced with their corresponding
      company names.
    - This dataset allows for a more interpretable presentation of return time
      series data.
    
    ---------------------------------------------------------------------------
    
                                ret_pivot
    
    This dataset is a filtered version of `ret_panel`, removing the row
    containing transformation codes.
    
    - The dataset retains only the time series data of excess returns.
    - The first column is reformatted to be a proper datetime index.
    - It is primarily used in return-based analyses that do not require
      transformation codes.
    
    ---------------------------------------------------------------------------
    
                                aug_ret_pivot
    
    This dataset augments `ret_pivot` by including one-period lagged excess
    returns for each stock.
    
    - The lagged excess returns are computed by shifting the original dataset
      by one period.
    - Lagged columns are renamed with a `_lag1` suffix for clarity.
    - The first row, containing NaN values due to lagging, is dropped.
    - This dataset enables analyses that require historical return dependencies,
      such as autoregressive modeling.
    
    ---------------------------------------------------------------------------
    
                                aug_ret_panel
    
    This dataset extends `aug_ret_pivot` by incorporating transformation codes
    for estimation purposes.
    
    - A transformation row is inserted, indicating that all variables are
      directly used in estimation.
    - The structure mirrors `aug_ret_pivot`, except for the added 
      transformation row.
    - This dataset is specifically formatted for regression and machine 
      learning
      applications requiring predefined transformations.
    
    ---------------------------------------------------------------------------

    """
    ###############################################################################
    ######################### Load and manage Databases ###########################
    ###############################################################################
    
    ################################# crsp_m ######################################
    
    # Set dates 
    #start_date = pd.to_datetime("1926-01-31") # Entire CRSP sample
    #start_date = "1962-07-31" # AMEX birth. 
    start_date = pd.to_datetime("1973-01-31") # NASDAQ birth is 1972-12-31
    end_date = pd.to_datetime("2023-12-31")

    """
                              CPI data from FRED
    """
    
    cpi_m = (pdr.DataReader(name="CPIAUCNS", data_source="fred", 
                            start="1972-12-31", end=end_date)
                .reset_index(names="month")
                .rename(columns={"CPIAUCNS": "cpi"})
                .assign(cpi=lambda x: x["cpi"]/x["cpi"].iloc[-1]))
    cpi_m['month'] = (cpi_m['month'].dt.to_period('M').dt.to_timestamp('M'))

    """
                      Risk-free rate (RF) from FF database
    """ 
    
    rf_m  = (pdr.DataReader(
      name="F-F_Research_Data_Factors",
      data_source="famafrench", 
      start="1972-12-31", 
      end=end_date)[0][['RF']].divide(100)
                              .reset_index(names="month")
                              .assign(month=lambda x: 
                                      pd.to_datetime(x["month"].astype(str)))
                              .rename(str.lower, axis="columns"))
    rf_m['month'] = (rf_m['month'].dt.to_period('M').dt.to_timestamp('M'))

    """ 
                               CRSP stock returns:
    """
    
    # Define the file path 
    file_path = "data/CRSP-Stock-Monthly.csv"
    
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
    for chunk in pd.read_csv(file_path, usecols=columns_to_keep, 
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
        chunk_filtered = chunk[chunk["exchcd"].isin([1, 31, 2, 32, 3, 33])]
        
        # Apply filters for ordinary common stocks (share codes 10 and 11)
        chunk_filtered = chunk_filtered[chunk_filtered["shrcd"].isin([10, 11])]
    
        # Exclude financial firms (SIC codes between 6000 and 6999)
        chunk_filtered = chunk_filtered[~chunk_filtered['siccd'].between(6000, 6999)]
        
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
    
    # Show stocks with the highest percentage of missing values
    #missing_data_summary_sorted.head(800)

    # The number of missing values, after adjust for delisting returns, is not
    # big in proportion to total observations. So below we just set 0 to these
    # missing values. 
    
    # Replace by 0 the the excess return that are missing:
    crsp_m['ret_excess'] = (crsp_m['ret_excess'].fillna(0))
        
    ###############################################################################
    
    ############################# Observed factors ################################
    
    # Market Factor from FF Library
    
    # MKT factor
    mkt_factor = (pdr.DataReader(
      name="F-F_Research_Data_Factors",
      data_source="famafrench", 
      start="1972-12-31", 
      end=end_date)[0][['Mkt-RF']].divide(100)
                              .reset_index(names="month")
                              .assign(month=lambda x: 
                                      pd.to_datetime(x["month"].astype(str)))
                              .rename(str.lower, axis="columns"))
    
    mkt_factor['month'] = (mkt_factor['month'].dt.to_period('M').dt.to_timestamp('M'))
    
    # Global Factor Data (GFD)
    
    # Observed factors:
    obs_F = pd.read_csv("data/[usa]_[all_factors]_[monthly]_[vw_cap].csv") 
    # Convert the 'date' column to datetime
    obs_F['date'] = pd.to_datetime(obs_F['date'])
    # Filter data starting from 1973-01-31
    obs_F = obs_F[obs_F['date'] >= '1973-03-31']
    # Set the 'date' column as the index
    #obs_F.set_index('date', inplace=True)
    # Transform into a pivot table
    obs_F = obs_F.pivot_table(index='date', columns='name', values='ret')
    
    # Now, we need to merge the market factor from FF library with the observed 
    # factors from Global Factor Data 
    
    mkt_factor['month'] = pd.to_datetime(mkt_factor['month'])
    obs_F = obs_F.reset_index()  # Temporarily reset the index
    obs_F['date'] = pd.to_datetime(obs_F['date'])
    
    # Merge the datasets on the date columns
    obs_F = obs_F.sort_values(by='date')
    merged_df = obs_F.merge(mkt_factor[['month', 'mkt-rf']], 
                            left_on='date', 
                            right_on='month', 
                            how='left')
    
    # Reorganize columns to place 'mkt-rf' immediately after 'date'
    columns = ['date', 'mkt-rf'] + [col for col in obs_F.columns if col != 'date']
    obs_F = merged_df[columns]
    # Set the 'date' column as the index
    obs_F.set_index('date', inplace=True)
    # Ensure the index is parsed as datetime
    obs_F.index = pd.to_datetime(obs_F.index)
        
    ###############################################################################
    
    ################################ ret_panel ####################################
    
    # BUILDING THE PANEL OF RETURNS (WITHOUT LAGS)
    
    # Pivot the DataFrame
    ret_panel = crsp_m.pivot_table(index='month', columns='permno', 
                                   values='ret_excess')
    # Ensure the date index is formatted as YYYY-MM-DD without time
    ret_panel.index = ret_panel.index.strftime('%Y-%m-%d')
    
    # Add a new row with transformation codes
    # Create a new row called "Transform:" with 1 in every column except the first
    transform_row = pd.DataFrame([[1]*len(ret_panel.columns)], 
                                 columns=ret_panel.columns)
    transform_row.index = ['Transform:']
    # Insert the new row after the first row
    ret_panel = pd.concat([transform_row, ret_panel])
    
    ################################ ret_pivot ####################################

    # It gives the panel of returns without the transformation codes 

    # Filters out rows where the first column's values are equal to "Transform:".
    # That is, it will remove the row with transformation codes.
    ret_pivot = ret_panel.drop(index="Transform:", errors="ignore")
    
    ###############################################################################
    
    ############################ ret_panel_comnam #################################
    
    # Creates a panel of returns with the company names as the header (instead of 
    # permno as the header)
    
    # Ensure 'month' is in datetime format for accurate sorting
    crsp_m['month'] = pd.to_datetime(crsp_m['month'])
    # Sort by 'permno' and 'month' in descending order
    crsp_m_sorted = crsp_m.sort_values(by=['permno', 'month'], 
                                       ascending=[True, False])
    # Drop duplicates to retain only the most recent 'comnam' for each 'permno'
    permno_to_comnam = (crsp_m_sorted
                        .drop_duplicates(subset='permno', 
                                         keep='first')[['permno', 'comnam']])
    # Set 'permno' as the index and extract 'comnam' for mapping
    permno_to_comnam = permno_to_comnam.set_index('permno')['comnam']
    # Replace permno columns in ret_panel with company names
    ret_panel_comnam = ret_pivot.rename(columns=permno_to_comnam)
        
    ###############################################################################
    
    ############################### aug_ret_pivot #################################
    
    # BUILIDING THE AUGMENTED PANEL OF RETURNS (WITH ONE-PERIOD LAGS) WITHOUT 
    # TRANSFORMATION CODES 
    
    # Create the lagged returns DataFrame
    lagged_returns = ret_pivot.shift(1)
    # Rename the lagged columns to indicate that they are lagged
    lagged_returns.columns = [f"{col}_lag1" for col in lagged_returns.columns]
    # Concatenate the original returns with the lagged returns
    aug_ret_pivot = pd.concat([ret_pivot, lagged_returns], axis=1)
    # Drop the first row, which corresponds to the NaNs from the lagging process
    aug_ret_pivot = aug_ret_pivot.iloc[1:]
        
    # BUILIDING THE AUGMENTED PANEL OF RETURNS (WITH ONE-PERIOD LAGS) WITH 
    # TRANSFORMATION CODES (FOR ESTIMATION)
    
    # Add a new row with transformation codes
    # Create a new row called "Transform:" with 1 in every column except the first
    transform_row = pd.DataFrame([[1]*len(aug_ret_pivot.columns)], 
                                 columns=aug_ret_pivot.columns)
    transform_row.index = ['Transform:']
    # Insert the new row after the first row
    aug_ret_panel = pd.concat([transform_row, aug_ret_pivot])