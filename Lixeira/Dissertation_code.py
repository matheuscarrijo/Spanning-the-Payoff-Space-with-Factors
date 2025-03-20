#!/usr/bin/env python
# coding: utf-8

# In[2]:


###############################################################################
############################ Importing Packages ###############################
###############################################################################

import numpy as np
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
import sqlite3
from datetime import datetime
import seaborn as sns
import os

# The following packages comes from .py functions stored in the same directory
import prepare_missing as pm
import remove_outliers as ro
import factors_em as fem
import mrsq

# Ignore some types of warnings
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# In[3]:


###############################################################################
######################### Load and manage Databases ###########################
###############################################################################

# Set dates 
start_date_1926 = pd.to_datetime("1926-01-31") # Entire sample
#start_date_1962 = "1962-07-31" # AMEX birth. Not used.
# Convert the start date to a datetime object
start_date_1973 = pd.to_datetime("1973-01-31") # NASDAQ birth is 1972-12-31
end_date = pd.to_datetime("2023-12-31") 

########################### CPI data from FRED ################################

cpi_m = (pdr.DataReader(name="CPIAUCNS", data_source="fred", 
                        start="1972-12-31", end=end_date)
            .reset_index(names="month")
            .rename(columns={"CPIAUCNS": "cpi"})
            .assign(cpi=lambda x: x["cpi"]/x["cpi"].iloc[-1]))

cpi_m['month'] = (cpi_m['month'].dt.to_period('M').dt.to_timestamp('M'))

################### Data about risk free from fama french database ############

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

##########################$$$ CRSP stock returns ##############################

# Define the file path
file_path = "CRSP-Stock-Monthly.csv"

# List of columns to keep (add 'DLRET' and 'DLSTCD' for delisting return 
# and delisting code)
columns_to_keep = [
    'date', 'PERMNO', 'SHRCD', 'RET', 'DLRET', 'DLSTCD', 'SHROUT', 'PRC',
    'EXCHCD', 'COMNAM', 'SICCD'
]
# Initialize an empty list to store results
data_list = []
# Define the chunk size
chunk_size = 100000

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

    # Filter the data to load only records starting from 'start_date_1973'
    chunk = chunk[chunk['month'] >= start_date_1973]

    # Apply filters for specific exchanges (NYSE, NASDAQ, AMEX)
    chunk_filtered = chunk[chunk["exchcd"].isin([1, 31, 2, 32, 3, 33])]
    
    # Apply filters for ordinary common stocks (share codes 10 and 11)
    chunk_filtered = chunk_filtered[chunk_filtered["shrcd"].isin([10, 11])]

    # Exclude financial firms (SIC codes between 6000 and 6999)
    chunk_filtered = chunk_filtered[~chunk_filtered['siccd'].between(6000, 6999)]
    
    # Adjust dlret for specific delisting codes and exchanges
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

# Integrate the risk-free rate from Fama-French three-factor model data 
# into the CRSP monthly data to calculate excess returns. The excess return is
# the return earned by a security or portfolio over the risk-free rate.
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


# In[6]:


###############################################################################
######################## Inspecting missing values ############################
###############################################################################

# MISSING DATA IN `crsp_m` DATASET

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


# In[4]:


# Replace by 0 the the excess return that are missing:
crsp_m['ret_excess'] = (crsp_m['ret_excess'].fillna(0))


# In[5]:


# BUILDING THE PANEL OF RETURNS (WITHOUT LAGS)

# Pivot the DataFrame
ret_pivot = crsp_m.pivot_table(index='month', columns='permno', values='ret_excess')
# Ensure the 'month' column is formatted as 'YYYY-MM-DD' without time
ret_pivot.index = ret_pivot.index.strftime('%Y-%m-%d')

# Add a new row with transformation codes
# Create a new row called "Transform:" with 1 in every column except the first
transform_row = pd.DataFrame([[1]*len(ret_pivot.columns)], columns=ret_pivot.columns)
transform_row.index = ['Transform:']
# Insert the new row after the first row
ret_panel = pd.concat([transform_row, ret_pivot])

# Save the ret_panel DataFrame to CSV
ret_panel.to_csv("ret_panel.csv")


# In[6]:


# VISUALIZING THE UNBALANCED PANEL `ret_pivot` 

# Create a binary matrix where True represents missing values and False otherwise
missing_data = ret_pivot.isnull()
custom_cmap = ListedColormap(['black', 'white'])
# Create the plot
plt.figure(figsize=(20, 5))  # Adjust the figure size for better visualization
# Use imshow to create a heatmap-like plot
plt.imshow(missing_data, aspect='auto', cmap=custom_cmap, interpolation='none')
# Set labels and title
plt.xlabel('Securities')
plt.ylabel('Time')
plt.title('Missing Data Pattern Across Securities Over Time')
# Invert y-axis so that earlier dates are at the top
plt.gca().invert_yaxis()
# Show the plot
plt.show()


# In[7]:


# BUILIDING THE AUGMENTED PANEL OF RETURNS (WITH ONE-PERIOD LAGS)

# Create the lagged returns DataFrame
lagged_returns = ret_pivot.shift(1)
# Rename the lagged columns to indicate that they are lagged
lagged_returns.columns = [f"{col}_lag1" for col in lagged_returns.columns]
# Concatenate the original returns with the lagged returns
aug_ret_pivot = pd.concat([ret_pivot, lagged_returns], axis=1)
# Drop the first row, which corresponds to the NaNs from the lagging process
aug_ret_pivot = aug_ret_pivot.iloc[1:]

# Add a new row with transformation codes
# Create a new row called "Transform:" with 1 in every column except the first
transform_row = pd.DataFrame([[1]*len(aug_ret_pivot.columns)], 
                             columns=aug_ret_pivot.columns)
transform_row.index = ['Transform:']
# Insert the new row after the first row
aug_ret_panel = pd.concat([transform_row, aug_ret_pivot])

# Save the ret_panel DataFrame to CSV
aug_ret_panel.to_csv("aug_ret_panel.csv")


# In[8]:


###############################################################################
########### Factors estimation using PCA as in Stock and Watson (2002) ########
###############################################################################

# Obs.: List of auxiliary functions need to be saved in same folder as this 
# script.

#                   ESTIMATION FOR PANEL WITHOUT LAGS                         # 

########################## Set Parameters #####################################

ret_panel = 'ret_panel.csv' # Panel of returns 

# Type of transformation performed on each series before factors are estimated
#   0 --> no transformation
#   1 --> demean only
#   2 --> demean and standardize
#   3 --> recursively demean and then standardize
DEMEAN = 2

# Information criterion used to select number of factors; for more details,
# see auxiliary function factors_em()
#   1 --> information criterion PC_p1
#   2 --> information criterion PC_p2
#   3 --> information criterion PC_p3
jj = 2

# Maximum number of factors to be estimated; if set to 99, the number of
# factors selected is forced to equal 8
kmax = 10

####################### PART 1: LOAD AND LABEL DATA ###########################

# Load data from CSV file and rename column
dum = pd.read_csv(ret_panel).dropna(how='all')       
dum.rename(columns={'Unnamed: 0': 'month'}, inplace=True)

series = dum.columns.values     # Variable names
tcode = dum.iloc[0, :]          # Transformation numbers
rawdata = dum.iloc[1:, :]       # Raw data
rawdata.set_index('month', inplace=True, drop=True)
rawdata.index.name = 'date'
T = len(rawdata)                # T = number of months in sample


######################## PART 2: PROCESS DATA #################################

# Transform raw data to be stationary using auxiliary function & 
# prepare_missing()
yt = pm.prepare_missing(rawdata, tcode)

# Reduce sample to usable dates: remove first two months because some
# series have been first differenced
yt = yt.iloc[2:,:]

# Remove outliers using auxiliary function remove_outliers();
#     data = matrix of transformed series with outliers removed
#     n = number of outliers removed from each series
data, n = ro.remove_outliers(yt)
#data = yt


 ############## PART 3: ESTIMATE FACTORS AND COMPUTE R-SQUARED ################

 # Estimate factors using function factors_em()
 #   pred    = Predicted values of the dataset based on the estimated factors.
 #   ehat    = difference between data and values of data predicted by the
 #             factors
 #   Fhat    = set of factors
 #   lamhat  = factor loadings
 #   ve2     = eigenvalues of data'*data
 #   x2      = data with missing values replaced from the EM algorithm

pred, ehat, Fhat, lamhat, ve2, x2 = fem.factors_em(data, kmax, jj, DEMEAN)

Fhat = pd.DataFrame(Fhat, index = data.index)
ehat = pd.DataFrame(ehat, index = data.index)
pred = pd.DataFrame(pred, index = data.index)
x2 = pd.DataFrame(x2, index = data.index)
lamhat = pd.DataFrame(lamhat, index = data.columns)
ve2 = pd.DataFrame(ve2)

Fhat.to_csv('output/factors.csv')
ehat.to_csv('output/ehat.csv')
pred.to_csv('output/pred.csv')
x2.to_csv('output/x2.csv')
lamhat.to_csv('output/lamhat.csv')
ve2.to_csv('output/ve2.csv')

#  Compute R-squared and marginal R-squared from estimated factors and
#  factor loadings using function mrsq()
#    R2      = R-squared for each series for each factor
#    mR2     = marginal R-squared for each series for each factor
#    mR2_F   = marginal R-squared for each factor
#    R2_T    = total variation explained by all factors
#    t10_s   = top 10 series that load most heavily on each factor
#    t10_mR2 = marginal R-squared corresponding to top 10 series
#              that load most heavily on each factor
#
#
# R2, mR2, mR2_F, R2_T, t10_s, t10_mR2 = mrsq.mrsq(Fhat,lamhat,ve2,data.columns.values)


# In[14]:


# Show desired results 
#print('R2', pd.DataFrame(R2).to_string())
#print('mR2', pd.DataFrame(mR2).to_string())
#print('mR2_F', mR2_F)
#print('R2_T', R2_T)
#print('t10_s', pd.DataFrame(t10_s).to_string())
#print('t10_mR2', pd.DataFrame(t10_mR2).to_string())


# In[9]:


# Plot each factor's time series with sequential x-axis
num_factors = Fhat.shape[1]
plt.figure(figsize=(15, 5 * num_factors))

# Create a range for the x-axis (1 to 612)
x_values = np.arange(1, len(Fhat) + 1)

for i in range(num_factors):
    plt.subplot(num_factors, 1, i+1)
    plt.plot(x_values, Fhat.iloc[:, i], label=f'Factor {i+1}')
    plt.title(f'Time Series of Factor {i+1}')
    plt.xlabel('Observation Number')
    plt.ylabel('Factor Value')
    plt.legend(loc='upper right')
    plt.grid(True)
    #Set x-ticks at regular intervals to reduce clutter
    tick_interval = 50  # Adjust this value as needed
    plt.xticks(ticks=np.arange(0, len(Fhat)+1, tick_interval))

plt.tight_layout()
plt.show()


# In[12]:


###############################################################################
########### Factors estimation using PCA as in Stock and Watson (2002) ########
###############################################################################

# Obs.: List of auxiliary functions need to be saved in same folder as this 
# script.

#                ESTIMATION FOR PANEL WITHOUT LAGS with kmax=5                # 

########################## Set Parameters #####################################

ret_panel = 'ret_panel.csv' # Panel of returns 

# Type of transformation performed on each series before factors are estimated
#   0 --> no transformation
#   1 --> demean only
#   2 --> demean and standardize
#   3 --> recursively demean and then standardize
DEMEAN = 2

# Information criterion used to select number of factors; for more details,
# see auxiliary function factors_em()
#   1 --> information criterion PC_p1
#   2 --> information criterion PC_p2
#   3 --> information criterion PC_p3
jj = 2

# Maximum number of factors to be estimated; if set to 99, the number of
# factors selected is forced to equal 8
kmax = 5

####################### PART 1: LOAD AND LABEL DATA ###########################

# Load data from CSV file and rename column
dum = pd.read_csv(ret_panel).dropna(how='all')       
dum.rename(columns={'Unnamed: 0': 'month'}, inplace=True)

series = dum.columns.values     # Variable names
tcode = dum.iloc[0, :]          # Transformation numbers
rawdata = dum.iloc[1:, :]       # Raw data
rawdata.set_index('month', inplace=True, drop=True)
rawdata.index.name = 'date'
T = len(rawdata)                # T = number of months in sample


######################## PART 2: PROCESS DATA #################################

# Transform raw data to be stationary using auxiliary function & 
# prepare_missing()
yt = pm.prepare_missing(rawdata, tcode)

# Reduce sample to usable dates: remove first two months because some
# series have been first differenced
yt = yt.iloc[2:,:]

# Remove outliers using auxiliary function remove_outliers();
#     data = matrix of transformed series with outliers removed
#     n = number of outliers removed from each series
data, n = ro.remove_outliers(yt)
#data = yt


 ############## PART 3: ESTIMATE FACTORS AND COMPUTE R-SQUARED ################

 # Estimate factors using function factors_em()
 #   pred    = Predicted values of the dataset based on the estimated factors.
 #   ehat    = difference between data and values of data predicted by the
 #             factors
 #   Fhat    = set of factors
 #   lamhat  = factor loadings
 #   ve2     = eigenvalues of data'*data
 #   x2      = data with missing values replaced from the EM algorithm

pred5, ehat5, Fhat5, lamhat5, ve2_5, x2_5 = fem.factors_em(data, kmax, jj, DEMEAN)

Fhat5 = pd.DataFrame(Fhat5, index = data.index)
ehat5 = pd.DataFrame(ehat5, index = data.index)
pred5 = pd.DataFrame(pred5, index = data.index)
x2_5 = pd.DataFrame(x2_5, index = data.index)
lamhat5 = pd.DataFrame(lamhat5, index = data.columns)
ve2_5 = pd.DataFrame(ve2_5)

Fhat5.to_csv('output/factors5.csv')
ehat5.to_csv('output/ehat5.csv')
pred5.to_csv('output/pred5.csv')
x2_5.to_csv('output/x2_5.csv')
lamhat5.to_csv('output/lamhat5.csv')
ve2_5.to_csv('output/ve2_5.csv')

#  Compute R-squared and marginal R-squared from estimated factors and
#  factor loadings using function mrsq()
#    R2      = R-squared for each series for each factor
#    mR2     = marginal R-squared for each series for each factor
#    mR2_F   = marginal R-squared for each factor
#    R2_T    = total variation explained by all factors
#    t10_s   = top 10 series that load most heavily on each factor
#    t10_mR2 = marginal R-squared corresponding to top 10 series
#              that load most heavily on each factor
#
#
# R2, mR2, mR2_F, R2_T, t10_s, t10_mR2 = mrsq.mrsq(Fhat,lamhat,ve2,data.columns.values)


# In[13]:


# Plot each factor's time series with sequential x-axis
num_factors = Fhat5.shape[1]
plt.figure(figsize=(15, 5 * num_factors))

# Create a range for the x-axis (1 to 612)
x_values = np.arange(1, len(Fhat5) + 1)

for i in range(num_factors):
    plt.subplot(num_factors, 1, i+1)
    plt.plot(x_values, Fhat5.iloc[:, i], label=f'Factor {i+1}')
    plt.title(f'Time Series of Factor {i+1}')
    plt.xlabel('Observation Number')
    plt.ylabel('Factor Value')
    plt.legend(loc='upper right')
    plt.grid(True)
    #Set x-ticks at regular intervals to reduce clutter
    tick_interval = 50  # Adjust this value as needed
    plt.xticks(ticks=np.arange(0, len(Fhat5)+1, tick_interval))

plt.tight_layout()
plt.show()


# In[9]:


###############################################################################
########### Factors estimation using PCA as in Stock and Watson (2002) ########
###############################################################################

# Obs.: List of auxiliary functions need to be saved in same folder as this 
# script.

#                  ESTIMATION FOR AUGMENTED PANEL (WITH LAGS)                 # 

########################## Set Parameters #####################################

aug_ret_panel = 'aug_ret_panel.csv' # Panel of returns 

# Type of transformation performed on each series before factors are estimated
#   0 --> no transformation
#   1 --> demean only
#   2 --> demean and standardize
#   3 --> recursively demean and then standardize
DEMEAN = 2

# Information criterion used to select number of factors; for more details,
# see auxiliary function factors_em()
#   1 --> information criterion PC_p1
#   2 --> information criterion PC_p2
#   3 --> information criterion PC_p3
jj = 2

# Maximum number of factors to be estimated; if set to 99, the number of
# factors selected is forced to equal 8
kmax = 15

####################### PART 1: LOAD AND LABEL DATA ###########################

# Load data from CSV file and rename column
dum = pd.read_csv(aug_ret_panel).dropna(how='all')       
dum.rename(columns={'Unnamed: 0': 'month'}, inplace=True)

series = dum.columns.values     # Variable names
tcode = dum.iloc[0, :]          # Transformation numbers
rawdata = dum.iloc[1:, :]       # Raw data
rawdata.set_index('month', inplace=True, drop=True)
rawdata.index.name = 'date'
T = len(rawdata)                # T = number of months in sample

######################## PART 2: PROCESS DATA #################################

# Transform raw data to be stationary using auxiliary function & 
# prepare_missing()
yt = pm.prepare_missing(rawdata, tcode)

# Reduce sample to usable dates: remove first two months because some
# series have been first differenced
yt = yt.iloc[2:,:]

# Remove outliers using auxiliary function remove_outliers();
#     data = matrix of transformed series with outliers removed
#     n = number of outliers removed from each series
data, n = ro.remove_outliers(yt)
#data = yt


 ############## PART 3: ESTIMATE FACTORS AND COMPUTE R-SQUARED ################

 # Estimate factors using function factors_em()
 #   pred    = Predicted values of the dataset based on the estimated factors.
 #   ehat    = difference between data and values of data predicted by the
 #             factors
 #   Fhat    = set of factors
 #   lamhat  = factor loadings
 #   ve2     = eigenvalues of data'*data
 #   x2      = data with missing values replaced from the EM algorithm

(aug_pred, aug_ehat, aug_Fhat, 
 aug_lamhat, aug_ve2, aug_x2) = fem.factors_em(data, kmax, jj, DEMEAN)

aug_Fhat = pd.DataFrame(aug_Fhat, index = data.index)
aug_ehat = pd.DataFrame(aug_ehat, index = data.index)
aug_pred = pd.DataFrame(aug_pred, index = data.index)
aug_x2 = pd.DataFrame(aug_x2, index = data.index)
aug_lamhat = pd.DataFrame(aug_lamhat, index = data.columns)
aug_ve2 = pd.DataFrame(aug_ve2) 

aug_Fhat.to_csv('output/aug_factors.csv')
aug_ehat.to_csv('output/aug_ehat.csv')
aug_pred.to_csv('output/aug_pred.csv')
aug_x2.to_csv('output/aug_x2.csv')
aug_lamhat.to_csv('output/aug_lamhat.csv')
aug_ve2.to_csv('output/aug_ve2.csv')

#  Compute R-squared and marginal R-squared from estimated factors and
#  factor loadings using function mrsq()
#    R2      = R-squared for each series for each factor
#    mR2     = marginal R-squared for each series for each factor
#    mR2_F   = marginal R-squared for each factor
#    R2_T    = total variation explained by all factors
#    t10_s   = top 10 series that load most heavily on each factor
#    t10_mR2 = marginal R-squared corresponding to top 10 series
#              that load most heavily on each factor
#
#
#(aug_R2, aug_mR2, aug_mR2_F, aug_R2_T, aug_t10_s, aug_t10_mR2 = 
# mrsq.mrsq(aug_Fhat, aug_lamhat, aug_ve2, data.columns.values))


# In[11]:


# Plot each factor's time series with sequential x-axis
num_factors = aug_Fhat.shape[1]
plt.figure(figsize=(15, 5 * num_factors))

# Create a range for the x-axis (1 to 612)
x_values = np.arange(1, len(aug_Fhat) + 1)

for i in range(num_factors):
    plt.subplot(num_factors, 1, i+1)
    plt.plot(x_values, aug_Fhat.iloc[:, i], label=f'Factor {i+1}')
    plt.title(f'Time Series of Factor {i+1}')
    plt.xlabel('Observation Number')
    plt.ylabel('Factor Value')
    plt.legend(loc='upper right')
    plt.grid(True)
    #Set x-ticks at regular intervals to reduce clutter
    tick_interval = 50  # Adjust this value as needed
    plt.xticks(ticks=np.arange(0, len(aug_Fhat)+1, tick_interval))

plt.tight_layout()
plt.show()


# In[15]:


###############################################################################
########### Factors estimation using PCA as in Stock and Watson (2002) ########
###############################################################################

# Obs.: List of auxiliary functions need to be saved in same folder as this 
# script.

#                  ESTIMATION FOR AUGMENTED PANEL (WITH LAGS)                 # 

########################## Set Parameters #####################################

aug_ret_panel = 'aug_ret_panel.csv' # Panel of returns 

# Type of transformation performed on each series before factors are estimated
#   0 --> no transformation
#   1 --> demean only
#   2 --> demean and standardize
#   3 --> recursively demean and then standardize
DEMEAN = 2

# Information criterion used to select number of factors; for more details,
# see auxiliary function factors_em()
#   1 --> information criterion PC_p1
#   2 --> information criterion PC_p2
#   3 --> information criterion PC_p3
jj = 2

# Maximum number of factors to be estimated; if set to 99, the number of
# factors selected is forced to equal 8
kmax = 5

####################### PART 1: LOAD AND LABEL DATA ###########################

# Load data from CSV file and rename column
dum = pd.read_csv(aug_ret_panel).dropna(how='all')       
dum.rename(columns={'Unnamed: 0': 'month'}, inplace=True)

series = dum.columns.values     # Variable names
tcode = dum.iloc[0, :]          # Transformation numbers
rawdata = dum.iloc[1:, :]       # Raw data
rawdata.set_index('month', inplace=True, drop=True)
rawdata.index.name = 'date'
T = len(rawdata)                # T = number of months in sample

######################## PART 2: PROCESS DATA #################################

# Transform raw data to be stationary using auxiliary function & 
# prepare_missing()
yt = pm.prepare_missing(rawdata, tcode)

# Reduce sample to usable dates: remove first two months because some
# series have been first differenced
yt = yt.iloc[2:,:]

# Remove outliers using auxiliary function remove_outliers();
#     data = matrix of transformed series with outliers removed
#     n = number of outliers removed from each series
data, n = ro.remove_outliers(yt)
#data = yt


 ############## PART 3: ESTIMATE FACTORS AND COMPUTE R-SQUARED ################

 # Estimate factors using function factors_em()
 #   pred    = Predicted values of the dataset based on the estimated factors.
 #   ehat    = difference between data and values of data predicted by the
 #             factors
 #   Fhat    = set of factors
 #   lamhat  = factor loadings
 #   ve2     = eigenvalues of data'*data
 #   x2      = data with missing values replaced from the EM algorithm

(aug_pred5, aug_ehat5, aug_Fhat5, 
 aug_lamhat5, aug_ve2_5, aug_x2_5) = fem.factors_em(data, kmax, jj, DEMEAN)

aug_Fhat5 = pd.DataFrame(aug_Fhat5, index = data.index)
aug_ehat5 = pd.DataFrame(aug_ehat5, index = data.index)
aug_pred5 = pd.DataFrame(aug_pred5, index = data.index)
aug_x2_5 = pd.DataFrame(aug_x2_5, index = data.index)
aug_lamhat5 = pd.DataFrame(aug_lamhat5, index = data.columns)
aug_ve2_5 = pd.DataFrame(aug_ve2_5) 

aug_Fhat5.to_csv('output/aug_factors5.csv')
aug_ehat5.to_csv('output/aug_ehat5.csv')
aug_pred5.to_csv('output/aug_pred5.csv')
aug_x2_5.to_csv('output/aug_x2_5.csv')
aug_lamhat5.to_csv('output/aug_lamhat5.csv')
aug_ve2_5.to_csv('output/aug_ve2_5.csv')

#  Compute R-squared and marginal R-squared from estimated factors and
#  factor loadings using function mrsq()
#    R2      = R-squared for each series for each factor
#    mR2     = marginal R-squared for each series for each factor
#    mR2_F   = marginal R-squared for each factor
#    R2_T    = total variation explained by all factors
#    t10_s   = top 10 series that load most heavily on each factor
#    t10_mR2 = marginal R-squared corresponding to top 10 series
#              that load most heavily on each factor
#
#
#(aug_R2, aug_mR2, aug_mR2_F, aug_R2_T, aug_t10_s, aug_t10_mR2 = 
# mrsq.mrsq(aug_Fhat, aug_lamhat, aug_ve2, data.columns.values))


# In[18]:


# Plot each factor's time series with sequential x-axis
num_factors = aug_Fhat5.shape[1]
plt.figure(figsize=(15, 5 * num_factors))

# Create a range for the x-axis (1 to 612)
x_values = np.arange(1, len(aug_Fhat5) + 1)

for i in range(num_factors):
    plt.subplot(num_factors, 1, i+1)
    plt.plot(x_values, aug_Fhat5.iloc[:, i], label=f'Factor {i+1}')
    plt.title(f'Time Series of Factor {i+1}')
    plt.xlabel('Observation Number')
    plt.ylabel('Factor Value')
    plt.legend(loc='upper right')
    plt.grid(True)
    #Set x-ticks at regular intervals to reduce clutter
    tick_interval = 50  # Adjust this value as needed
    plt.xticks(ticks=np.arange(0, len(aug_Fhat5)+1, tick_interval))

plt.tight_layout()
plt.show()

