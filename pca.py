import pandas as pd
from pca_aux.prepare_missing import prepare_missing as pm
from pca_aux.remove_outliers import remove_outliers as ro
from pca_aux.factors_em import factors_em as fem
from pca_aux.mrsq import mrsq

def pca_stock_watson(data, kmax=99, jj=2, DEMEAN=2):
    """
    Estimate factors using Principal Component Analysis (PCA) following 
    Stock and Watson (2002).

    Parameters:
    ----------
    data : str or pd.DataFrame
        - If a string, it should be a path to a CSV file containing the dataset.
        - If a DataFrame, it should already contain the data.
    kmax : int
        Maximum number of factors to estimate. If set to 99, the number of
        factors selected is forced to be 8.
    jj : int, optional (default=2)
        Information criterion used to select the number of factors:
        - 1: PC_p1
        - 2: PC_p2
        - 3: PC_p3
    DEMEAN : int, optional (default=2)
        Transformation applied to the data before estimating factors:
        - 0: No transformation
        - 1: Demean only
        - 2: Demean and standardize
        - 3: Recursively demean and then standardize

    Returns:
    -------
    tuple
        Contains:
        - pred (pd.DataFrame): Predicted values
        - ehat (pd.DataFrame): Residuals
        - Fhat (pd.DataFrame): Estimated factors
        - lamhat (pd.DataFrame): Factor loadings
        - ve2 (pd.DataFrame): Eigenvalues
        - x2 (pd.DataFrame): Data with missing values replaced by EM algorithm
        - R2 (pd.DataFrame): R-squared for each series for each factor
        - mR2 (pd.DataFrame): Marginal R-squared for each series
        - mR2_F (pd.DataFrame): Marginal R-squared for each factor
        - R2_T (pd.DataFrame): Total variation explained by all factors
        - t10_s (list): Top 10 series that load most heavily on each factor
        - t10_mR2 (pd.DataFrame): Marginal R-squared for top 10 series
    """

    ####################### PART 1: LOAD AND LABEL DATA #######################

    if isinstance(data, str):  # If `data` is a file path
        dum = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):  # If `data` is already a DataFrame
        dum = data.copy()
    else:
        raise ValueError("`data` must be either a file path (str) or a Pandas DataFrame.")

    dum = dum.dropna(how='all') # Remove empty rows

    # Ensure first column is properly named
    if 'Unnamed: 0' in dum.columns:
        dum.rename(columns={'Unnamed: 0': 'month'}, inplace=True)

    series = dum.columns.values  # Variable names
    tcode = dum.iloc[0, :]       # Transformation codes
    rawdata = dum.iloc[1:, :]    # Data excluding first row (transformation 
                                 # codes)

    # Ensure 'month' is the index
    if 'month' in rawdata.columns:
        rawdata.set_index('month', inplace=True, drop=True)

    rawdata.index.name = 'date'

    ######################## PART 2: PROCESS DATA #############################

    # Transform data using auxiliary function
    yt = pm(rawdata, tcode)

    # Reduce sample to usable dates: remove first one or two months because
    # some series have been first or second differenced
    if any(x in tcode for x in [2, 5]):  # Check if 2 or 5 is in tcode
        yt = yt.iloc[1:, :]  # Drop first row
        if any(x in tcode for x in [3, 4, 6, 7]):  # Check if 3, 4, 6, or 7 is 
                                                   # in tcode
            yt = yt.iloc[1:, :]  # Drop second row (since it's differentiated 
                                 # twice)


    # Remove outliers
    data, _ = ro(yt)

    ############## PART 3: ESTIMATE FACTORS AND COMPUTE R-SQUARED #############

    # Estimate factors
    pred, ehat, Fhat, lamhat, ve2, x2 = fem(data, kmax, jj, DEMEAN)

    # Convert results to DataFrames
    pred = pd.DataFrame(pred, index=data.index)
    ehat = pd.DataFrame(ehat, index=data.index)
    Fhat = pd.DataFrame(Fhat, index=data.index)
    lamhat = pd.DataFrame(lamhat, index=data.columns)
    ve2 = pd.DataFrame(ve2)
    x2 = pd.DataFrame(x2, index=data.index)

    # Compute R-squared and marginal R-squared
    #R2, mR2, mR2_F, R2_T, t10_s, t10_mR2 = mrsq(Fhat, lamhat, ve2, 
    #                                            data.columns.values)

    return pred, ehat, Fhat, lamhat, ve2, x2
    #, R2, mR2, mR2_F, R2_T, t10_s, t10_mR2