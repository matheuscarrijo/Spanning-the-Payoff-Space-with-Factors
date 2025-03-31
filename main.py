#======================
# IMPORTING PACKAGES
#======================

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import seaborn as sns
import os

# The folowing packages must be in your directory
from pca import pca_stock_watson
from growl import growl_fista
from data import crsp_m, gfd, return_panels
from cross_validation import grid_search_CV, cross_validate_growl

#=============================================
# GET ASSET RETURNS AND OBSERVED FACTORS  
#=============================================

# Step 1: Load CRSP data
crsp_m = crsp_m(data_path="data/CRSP-Stock-Monthly.csv",
                start_date="1973-01-31",
                end_date="2023-12-31")
# Step 2: Load observed factors (Global Factor Data + Market from FF database)
obs_f_gfd = gfd(gfd_path="data/[usa]_[all_factors]_[monthly]_[vw_cap].csv",
                  start_date="1973-01-31",
                  end_date="2023-12-31")
# Step 3: Build return panels from CRSP data
ret_panel, ret_panel_comnam, aug_ret_panel = return_panels(crsp_m)

# Save in csv format

#crsp_m.to_csv("data/crsp_m.csv")
#obs_f_gfd.to_csv('data/obs_f_gfd.csv')
#ret_panel.to_csv("data/ret_panel.csv")
#ret_panel_comnam.to_csv("data/ret_panel_comnam.csv")
#aug_ret_panel.to_csv("data/aug_ret_panel.csv")
#ret_pivot.to_csv("data/ret_pivot.csv")
#aug_ret_pivot.to_csv("data/aug_ret_pivot.csv")

# If you already have the required datasets, just upload:
    
# crsp_m = pd.read_csv("data/crsp_m.csv", index_col=0)
# obs_f_gfd = pd.read_csv("data/obs_f_gfd.csv", index_col=0)
# ret_panel = pd.read_csv("data/ret_panel.csv", index_col=0)
# ret_panel_comnam = pd.read_csv("data/ret_panel_comnam.csv", index_col=0)
# aug_ret_panel = pd.read_csv("data/aug_ret_panel.csv", index_col=0)
# ret_pivot = pd.read_csv("data/ret_pivot.csv", index_col=0)
# aug_ret_pivot = pd.read_csv("data/aug_ret_pivot.csv", index_col=0)

#============================================================
# FACTORS ESTIMATION USING PCA AS IN Stock and Watson (2002) 
#============================================================

# ESTIMATION WITHOUT LAGS
# ret_panel = "data/ret_panel.csv"
jj = 2 # an integer indicating the information criterion used for selecting the 
       # number of factors; it can take on  the following values:
       #         1 (information criterion PC_p1)
       #         2 (information criterion PC_p2)
       #         3 (information criterion PC_p3)       
       # This information criterions comes from Bai and Ng (2002)  
DEMEAN = 2 # an integer indicating the type of transformation performed on each 
           # series in 'ret_panel' before the factors are estimated; it can 
           # take on the following values:
           #     0 (no transformation)
           #     1 (demean only)
           #     2 (demean and standardize)
           #     3 (recursively demean and then standardize) 
kmax = 10   # an integer indicating the maximum number of factors to be 
            # estimated; if set to 99, the number of factors selected is forced
            # to equal 8    
            
# results_ret_k10 is the list [pred, ehat, Fhat, lamhat, ve2, x2].
results_ret_k10 = pca_stock_watson(ret_panel, kmax=kmax, jj=jj, DEMEAN=DEMEAN)
# kmax = 5
# results_ret_k5 = pca_stock_watson(ret_panel, kmax=5, jj=jj, DEMEAN=DEMEAN)

# ESTIMATION WITH LAGS 
# aug_ret_panel = "data/aug_ret_panel.csv"
# kmax = 15
#results_aug_15 = pca_stock_watson(aug_ret_panel, kmax=15, jj=jj, DEMEAN=DEMEAN)
# results_aug_15 = [pred, ehat, Fhat, lamhat, ve2, x2] 
# kmax = 5
# results_aug_5 = pca_stock_watson(aug_ret_panel, kmax=5, jj=jj, DEMEAN=DEMEAN)

# SAVE OUTPUTS IN CSV FORMAT 
# Create output directory if it doesn't exist
#os.makedirs("outputs", exist_ok=True)
#results_ret_k10[0].to_csv("outputs/pred.csv") 
#results_ret_k10[3].to_csv("outputs/gamma_hat.csv")

# Load the data in the following way if you have the datasets stored 
#obs_f_gfd = pd.read_csv("output/obs_f_gfd.csv", index_col=0)

#PCs = pd.read_csv("output/pred.csv", index_col=0)
PCs = results_ret_k10[0]

#gamma_hat = pd.read_csv("output/lamhat.csv", index_col=0)
gamma_hat = results_ret_k10[3]

# Normalize the features (X) and targets (Y)
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
obs_f_gfd_normalized = scaler_X.fit_transform(obs_f_gfd)
PCs_normalized = scaler_Y.fit_transform(PCs)

#=============================
# CROSS VALIDATION FOR GROWL 
#=============================

# Define the hypter-parameters grids:
#lambda1_list_start = np.logspace(-2, 3, 6)  # i.e. 0.01, 0.1, 1, 10, 100, 1000
lambda1_list_start = np.logspace(1, 3, 3)
#lambda2_list_start = np.logspace(-2, 3, 6)
lambda2_list_start = np.logspace(-2, 0, 3)
#ramp_size_list_start = [0.1, 0.3, 0.5, 0.7, 0.9]
ramp_size_list_start = [0.5, 0.7, 0.9]

# Just to test the CV algorithm:
lambda1_list_start = np.logspace(1, 2, 2) 
lambda2_list_start = np.logspace(-2, -1, 2)
ramp_size_list_start = [0.7, 0.9]


# Run CV with the adaptive grid refinement approach
best_score, best_params, best_B, stages_info = grid_search_CV(
    obs_f_gfd_normalized, PCs_normalized,
    cross_validate_fn=cross_validate_growl,
    lambda1_list_start=lambda1_list_start,
    lambda2_list_start=lambda2_list_start,
    ramp_size_list_start=ramp_size_list_start,
    n_stages=5,        # up to 5 refinement stages
    n_splits=5,
    random_state=42,
    plot=True,
    verbose=True,
    min_improvement=1e-3  # or None if we don't want early stopping
)

print("Best params found:", best_params, "with CV MSE:", best_score)

#==========
# GrOWL 
#==========

# B_growl_oscar, _ = growl_fista(obs_f_gfd_normalized, PCs_normalized, w=None, 
#                           lambda_1=1.0, lambda_2=0.5, ramp_size=1, 
#                           max_iter=1000, tol=1e-4)

B_growl_oscar, _ = growl_fista(obs_f_gfd_normalized, PCs_normalized,
                            lambda_1=140,
                            lambda_2=1/3,
                            ramp_size=1, # Then, OSCAR-like regularization
                            max_iter=10000,
                            tol=1e-4,
                            check_type='solution_diff',
                            scale_objective=True,
                            verbose=True)

B_growl_oscar = pd.DataFrame(B_growl_oscar, index=obs_f_gfd.columns, 
                             columns=gamma_hat.index)

B_growl_oscar.to_csv("output/B_growl_oscar.csv")

# Transform in pandas dataframe without columns 
B_growl_oscar_without_zeros = (pd.DataFrame(B_growl_oscar.T, columns=obs_f_gfd.columns, 
                              index=gamma_hat.index)
                              .loc[:, (pd.DataFrame(B_growl_oscar.T, 
                                       columns=obs_f_gfd.columns) != 0)
                              .any(axis=0)])

# Heatmap of estimated B
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
import pandas as pd 

df = pd.read_csv("output/B_growl_oscar.csv")
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

#==================== OTHER REGULARIZATION PROCEDURES ========================

#==========
# LASSO 
#==========

# mtl = linear_model.MultiTaskLasso(
#         alpha=8.2, 
#         fit_intercept=False, 
#         copy_X=True, 
#         max_iter=1000, 
#         tol=1e-04, 
#         warm_start=False, 
#         random_state=None, 
#         selection='cyclic')

# mtl.fit(obs_f_gfd_normalized, PCs_normalized)

#print(mtl.coef_)
#pd.DataFrame(mtl.coef_, columns=obs_f_gfd.columns)
# beta_mtl = (pd.DataFrame(mtl.coef_, columns=obs_f_gfd.columns,
#                          index=gamma_hat.index)
#                          .loc[:, (pd.DataFrame(mtl.coef_, 
#                                   columns=obs_f_gfd.columns,) != 0)
#                          .any(axis=0)])

#=========================
# MULTI-TASK ELASTIC NET
#=========================

# mten = linear_model.MultiTaskElasticNet(
#        alpha=27.6, # 27.5
#         l1_ratio=0.515, # 0.515
#         fit_intercept=False, 
#         copy_X=True, 
#         max_iter=1000, 
#         tol=1e-08, 
#         warm_start=False, 
#         random_state=None, 
#         selection='cyclic')

# mten.fit(obs_f_gfd_normalized, PCs_normalized)

#print(mtl.coef_)
#pd.DataFrame(mtl.coef_, columns=obs_f_gfd.columns)
# beta_mten = (pd.DataFrame(mten.coef_, columns=obs_f_gfd.columns, 
#                           index=gamma_hat.index)
#                           .loc[:, (pd.DataFrame(mten.coef_, 
#                                 columns=obs_f_gfd.columns) != 0)
#                           .any(axis=0)])