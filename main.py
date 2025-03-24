###############################################################################
############################ Importing Packages ###############################
###############################################################################

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import seaborn as sns
import os

from pca_func import pca
from growl_func import growl

# Ignore some types of warnings
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

###############################################################################
############## Defining some simple useful functions for this code ############
###############################################################################

def save_results(results, dataset_name, kmax, jj, DEMEAN):
    """Saves PCA results with a filename that includes all parameter values."""
    (Fhat, ehat, pred, x2, lamhat, ve2, R2, mR2, 
     mR2_F, R2_T, t10_s, t10_mR2) = results

    # Unique filename suffix based on all parameters
    filename_suffix = f"{dataset_name}_kmax{kmax}_jj{jj}_D{DEMEAN}"
    
    Fhat.to_csv(f"{output_dir}/factors_{filename_suffix}.csv")
    ehat.to_csv(f"{output_dir}/ehat_{filename_suffix}.csv")
    pred.to_csv(f"{output_dir}/pred_{filename_suffix}.csv")
    x2.to_csv(f"{output_dir}/x2_{filename_suffix}.csv")
    lamhat.to_csv(f"{output_dir}/lamhat_{filename_suffix}.csv")
    ve2.to_csv(f"{output_dir}/ve2_{filename_suffix}.csv")
    R2.to_csv(f"{output_dir}/R2_{filename_suffix}.csv")
    mR2.to_csv(f"{output_dir}/mR2_{filename_suffix}.csv")
    mR2_F.to_csv(f"{output_dir}/mR2_F_{filename_suffix}.csv")
    pd.DataFrame([t10_s]).to_csv(f"{output_dir}/t10_s_{filename_suffix}.csv", 
                                 index=False)
    t10_mR2.to_csv(f"{output_dir}/t10_mR2_{filename_suffix}.csv")

def load_pca_results(dataset_name, kmax, jj, DEMEAN, output_dir="output"):
    """
    Loads saved PCA results from CSV files.

    Parameters:
    ----------
    dataset_name : str
        Name of the dataset (e.g., "ret_panel", "aug_ret_panel").
    kmax : int
        Number of factors used in the estimation.
    jj : int
        Information criterion used for factor selection.
    DEMEAN : int
        Data transformation method applied before estimating factors.
    output_dir : str, optional (default="output")
        Directory where the CSV files were saved.

    Returns:
    -------
    tuple
        Contains:
        - Fhat (pd.DataFrame): Estimated factors
        - ehat (pd.DataFrame): Residuals
        - pred (pd.DataFrame): Predicted values
        - x2 (pd.DataFrame): Data with missing values replaced by EM algorithm
        - lamhat (pd.DataFrame): Factor loadings
        - ve2 (pd.DataFrame): Eigenvalues
        - R2 (pd.DataFrame): R-squared for each series for each factor
        - mR2 (pd.DataFrame): Marginal R-squared for each series
        - mR2_F (pd.DataFrame): Marginal R-squared for each factor
        - R2_T (pd.DataFrame): Total variation explained by all factors
        - t10_s (list): Top 10 series that load most heavily on each factor
        - t10_mR2 (pd.DataFrame): Marginal R-squared for top 10 series
    """
    
    # Construct filename pattern
    filename_suffix = f"{dataset_name}_kmax{kmax}_jj{jj}_D{DEMEAN}"
    
    # List of filenames
    file_paths = {
        "Fhat": f"{output_dir}/factors_{filename_suffix}.csv",
        "ehat": f"{output_dir}/ehat_{filename_suffix}.csv",
        "pred": f"{output_dir}/pred_{filename_suffix}.csv",
        "x2": f"{output_dir}/x2_{filename_suffix}.csv",
        "lamhat": f"{output_dir}/lamhat_{filename_suffix}.csv",
        "ve2": f"{output_dir}/ve2_{filename_suffix}.csv",
        "R2": f"{output_dir}/R2_{filename_suffix}.csv",
        "mR2": f"{output_dir}/mR2_{filename_suffix}.csv",
        "mR2_F": f"{output_dir}/mR2_F_{filename_suffix}.csv",
        "R2_T": f"{output_dir}/R2_T_{filename_suffix}.csv",
        "t10_s": f"{output_dir}/t10_s_{filename_suffix}.csv",
        "t10_mR2": f"{output_dir}/t10_mR2_{filename_suffix}.csv",
    }

    # Dictionary to store loaded data
    results = {}

    # Load files, checking if they exist
    for key, path in file_paths.items():
        if os.path.exists(path):
            if key == "t10_s":  # This was saved as a list, needs different handling
                results[key] = pd.read_csv(path, header=None).values.tolist()[0]
            else:
                results[key] = pd.read_csv(path, index_col=0)
        else:
            print(f"Warning: File not found {path}")

    # Unpack results safely, ensuring missing files don't break the return
    return (
        results.get("Fhat"), results.get("ehat"), results.get("pred"),
        results.get("x2"), results.get("lamhat"), results.get("ve2"),
        results.get("R2"), results.get("mR2"), results.get("mR2_F"),
        results.get("R2_T"), results.get("t10_s"), results.get("t10_mR2")
    )

###############################################################################
########################## Data preprocessing #################################
###############################################################################

    """
    The data preprocessing documentation is here: 
    
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
# ret_panel.index = pd.to_datetime(ret_panel.index).to_period('D')

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
ret_pivot = ret_panel.iloc[1:, :]
    
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

    ########################### Save in csv format ############################

# Save in csv format
crsp_m.to_csv("data/crsp_m.csv")
obs_F.to_csv('data/obs_F.csv')
ret_panel.to_csv("data/ret_panel.csv")
ret_panel_comnam.to_csv("data/ret_panel_comnam.csv")
aug_ret_panel.to_csv("data/aug_ret_panel.csv")
ret_pivot.to_csv("data/ret_pivot.csv")
aug_ret_pivot.to_csv("data/aug_ret_pivot.csv")

####################### just load the preprocessed data #######################

# # If you already have the required datasets, just upload:

# crsp_m = pd.read_csv("data/crsp_m.csv", index_col=0)
# obs_F = pd.read_csv("data/obs_F.csv", index_col=0)
# ret_panel = pd.read_csv("data/ret_panel.csv", index_col=0)
# ret_panel_comnam = pd.read_csv("data/ret_panel_comnam.csv", index_col=0)
# aug_ret_panel = pd.read_csv("data/aug_ret_panel.csv", index_col=0)
# ret_pivot = pd.read_csv("data/ret_pivot.csv", index_col=0)
# aug_ret_pivot = pd.read_csv("data/aug_ret_pivot.csv", index_col=0)

###############################################################################
########### Factors estimation using PCA as in Stock and Watson (2002) ########
###############################################################################

# Create output directory if it doesn't exist
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

######################## ESTIMATION WITHOUT LAGS ##############################

ret_panel = "data/ret_panel.csv"
jj = 2
DEMEAN = 2

# kmax = 10
results_ret_k10 = pca(ret_panel, kmax=10, jj=jj, DEMEAN=DEMEAN)
# kmax = 5
results_ret_k5 = pca(ret_panel, kmax=5, jj=jj, DEMEAN=DEMEAN)

# Save results 
save_results(results_ret_k10, "ret_panel", kmax=10, jj=jj, DEMEAN=DEMEAN)
save_results(results_ret_k5, "ret_panel", kmax=5, jj=jj, DEMEAN=DEMEAN)

######################### ESTIMATION WITH LAGS ################################

aug_ret_panel = "data/aug_ret_panel.csv"

# kmax = 15
results_aug_15 = pca(aug_ret_panel, kmax=15, jj=jj, DEMEAN=DEMEAN)
# results_aug_15 = [pred, ehat, Fhat, lamhat, ve2, x2] 

# kmax = 5
results_aug_5 = pca(aug_ret_panel, kmax=5, jj=jj, DEMEAN=DEMEAN)

# Save results
save_results(results_aug_15, "aug_ret_panel", kmax=15, jj=jj, DEMEAN=DEMEAN)
save_results(results_aug_5, "aug_ret_panel", kmax=5, jj=jj, DEMEAN=DEMEAN)

################## just load the datasets from PCA estimation #################

# # If you already have the required datasets, no need to do all the estimation 
# # again. Just upload:
# # Load results for ret_panel with kmax=10, jj=2, DEMEAN=2
# loaded_results = load_pca_results("ret_panel", kmax=10, jj=2, DEMEAN=2)

# # Unpack the results
# (Fhat, ehat, pred, x2, lamhat, ve2, R2, 
#  mR2, mR2_F, R2_T, t10_s, t10_mR2) = loaded_results

###############################################################################
######################## Regularization Procedures ############################
###############################################################################

# Load and normalizes data 
obs_F = pd.read_csv("output/obs_F.csv", index_col=0)
PCs = pd.read_csv("output/pred.csv", index_col=0)
gamma_hat = pd.read_csv("output/lamhat.csv", index_col=0)

# Normalize the features (X) and targets (Y)
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

obs_F_normalized = scaler_X.fit_transform(obs_F)
PCs_normalized = scaler_Y.fit_transform(PCs)

############################### LASSO #########################################

mtl = linear_model.MultiTaskLasso(
        alpha=8.2, 
        fit_intercept=False, 
        copy_X=True, 
        max_iter=1000, 
        tol=1e-04, 
        warm_start=False, 
        random_state=None, 
        selection='cyclic')

mtl.fit(obs_F_normalized, PCs_normalized)

#print(mtl.coef_)
#pd.DataFrame(mtl.coef_, columns=obs_F.columns)
beta_mtl = (pd.DataFrame(mtl.coef_, columns=obs_F.columns,
                         index=gamma_hat.index)
                         .loc[:, (pd.DataFrame(mtl.coef_, 
                                  columns=obs_F.columns,) != 0)
                         .any(axis=0)])

############################### Elastic Net ###################################

mten = linear_model.MultiTaskElasticNet(
        alpha=27.6, # 27.5
        l1_ratio=0.515, # 0.515
        fit_intercept=False, 
        copy_X=True, 
        max_iter=1000, 
        tol=1e-08, 
        warm_start=False, 
        random_state=None, 
        selection='cyclic')

mten.fit(obs_F_normalized, PCs_normalized)

#print(mtl.coef_)
#pd.DataFrame(mtl.coef_, columns=obs_F.columns)
beta_mten = (pd.DataFrame(mten.coef_, columns=obs_F.columns, 
                          index=gamma_hat.index)
                          .loc[:, (pd.DataFrame(mten.coef_, 
                                columns=obs_F.columns) != 0)
                          .any(axis=0)])

################################## GrOWL ######################################

# B_growl_oscar, _ = growl(obs_F_normalized, PCs_normalized, w=None, 
#                           lambda_1=1.0, lambda_2=0.5, ramp_size=1, 
#                           max_iter=1000, tol=1e-4)

B_growl_oscar, _ = growl(obs_F_normalized, PCs_normalized,
                         lambda_1=140,
                         lambda_2=1/3,
                         ramp_size=1, # Then, OSCAR-like regularization
                         max_iter=10000,
                         tol=1e-4,
                         check_type='solution_diff',
                         scale_objective=True,
                         verbose=True)

B_growl_oscar = pd.DataFrame(B_growl_oscar, index=obs_F.columns, 
                              columns=gamma_hat.index)

B_growl_oscar.to_csv("output/B_growl_oscar.csv")

# Transform in pandas dataframe without zero columns
B_growl_oscar_without_zeros = (pd.DataFrame(B_growl_oscar.T, columns=obs_F.columns, 
                              index=gamma_hat.index)
                              .loc[:, (pd.DataFrame(B_growl_oscar.T, 
                                       columns=obs_F.columns) != 0)
                              .any(axis=0)])

# Heatmap of beta_estimated
plt.figure(figsize=(12, 8))
sns.heatmap(B_growl_oscar, cmap="coolwarm", 
            annot=False, cbar=True)
#plt.title("Heatmap of Factor loadings Across Firms (Multi-Task LASSO)", 
#          fontsize=16)
plt.xlabel("Observed Factors", fontsize=16)
plt.ylabel("Firms (permno)", fontsize=16)
#plt.savefig('Imgs/.pdf', 
#            dpi=300, bbox_inches='tight')  
plt.show()

# Heatmap of beta_estimated withou mkt-rf (since it can affect the 
# visualization because it has big values compared to other factors)
plt.figure(figsize=(12, 8))
sns.heatmap(B_growl_oscar.iloc[:,1:], cmap="coolwarm", 
            annot=False, cbar=True)
#plt.title("Heatmap of Factor loadings Across Firms (Multi-Task LASSO)", 
#          fontsize=16)
plt.xlabel("Observed Factors", fontsize=16)
plt.ylabel("Firms (permno)", fontsize=16)
#plt.savefig('Imgs/.pdf', 
#            dpi=300, bbox_inches='tight')  
plt.show()


# Compute the correlation matrix
corr_matrix = B_growl_oscar.corr()
# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', 
            cbar=True, square=True, linewidths=.5)
#plt.title('Correlation Matrix of betas', fontsize=16)
plt.tight_layout()
plt.show()

################################# GROUPS ######################################
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram

df = pd.read_csv("B_growl_oscar.csv")
# df = B_growl_oscar

# Extract the coefficient matrix (excluding the first column, which contains row names)
coef_matrix = df.iloc[:, 1:].to_numpy()

# Compute the row norms (L2 norm for each row)
row_norms = np.linalg.norm(coef_matrix, axis=1)

# Perform hierarchical clustering on row norms
Z = linkage(row_norms.reshape(-1, 1), method='complete')

# Define the threshold for grouping
threshold = 1.0

# Assign cluster labels based on the threshold
cluster_labels = fcluster(Z, t=threshold, criterion='distance')

# Create a mapping of clusters
groups = {}
for idx, cluster in enumerate(cluster_labels):
    if cluster not in groups:
        groups[cluster] = []
    groups[cluster].append(row_norms[idx])

# Display grouped norms
groups

################################################

# Extract the row names from the dataset
row_names = df.iloc[:, 0].values

# Map row names to their corresponding groups
grouped_rows = {}
for idx, cluster in enumerate(cluster_labels):
    if cluster not in grouped_rows:
        grouped_rows[cluster] = []
    grouped_rows[cluster].append(row_names[idx])

# Display grouped row names
grouped_rows

################################################

# Iterate through each group and plot the heatmap
for group_id, factor_names in grouped_rows.items():
    # Select the corresponding rows from the dataset
    subset = df[df.iloc[:, 0].isin(factor_names)].set_index(df.columns[0])

    # Plot heatmap with flipped axes
    plt.figure(figsize=(12, 8))
    sns.heatmap(subset.T, cmap="coolwarm", annot=False, cbar=True)
    plt.ylabel("Firms (permno)", fontsize=16)
    plt.xlabel("Observed Factors", fontsize=16)
    plt.title(f"Heatmap of Factor Loadings - Group {group_id}", fontsize=16)
    plt.show()


############################## CROSS VALIDATION ###############################

from sklearn.model_selection import KFold

# Define the hypter-parameters grid:
lambda1_list = np.logspace(-4, 2, 7)  # [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
#lambda1_list = [50, 75, 100, 125, 150]
lambda1_list = [125, 150, ]

#ramp_delta_list = np.logspace(-2, 1, 4) # [1e-2, 1e-1, 1e0, 1e1]
ramp_delta_list = [0.5, 0.75, 1.0, 1.25, 1.5]

#ramp_size_list = [10, 50, 100, 150] 
ramp_size_list = [130, 140, 150, 160, 170]

best_score = np.inf  # or -np.inf if we measure R^2
best_params = None
best_B = None

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for lam1 in lambda1_list:
    for rd in ramp_delta_list:
        for rs in ramp_size_list:
            scores = []
            for train_index, val_index in kf.split(obs_F_normalized):
                X_train, X_val = obs_F_normalized[train_index], obs_F_normalized[val_index]
                Y_train, Y_val = PCs_normalized[train_index], PCs_normalized[val_index]
                
                B_candidate, _ = growl(X_train, Y_train,
                                       lambda_1=lam1,
                                       lambda_2=rs,
                                       ramp_size=rs,
                                       max_iter=100,
                                       tol=1e-2,
                                       check_type='solution_diff',
                                       scale_objective=True,
                                       verbose=False)
                
                # Predict on validation fold
                Y_pred = X_val @ B_candidate  # shape: (n_val, r)
                
                # Compute a metric, e.g. MSE or RMSE (we use MSE)
                val_mse = np.mean((Y_val - Y_pred)**2)
                scores.append(val_mse)
                
            avg_val_score = np.mean(scores)
            print(f"(λ1={lam1}, ramp_delta={rd}, ramp_size={rs}) -> CV MSE = {avg_val_score:.4f}")
            
            # Keep track of best
            if avg_val_score < best_score:
                best_score = avg_val_score
                best_params = (lam1, rd, rs)
                best_B = B_candidate.copy()

print("Best params found:", best_params, "with CV MSE:", best_score)

# Distribuição da norma