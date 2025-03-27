import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from growl import growl_fista

def cross_validate_growl(X, Y, lambda1_list, lambda2_list, ramp_size_list,
                         n_splits=5, random_state=42):
    """
    Performs 5-fold CV over the given grids of (lambda1, lambda2, ramp_size).
    Returns:
      best_score, best_params, best_B
      all_results: list of tuples ((lam1, lam2, rs), mse)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    best_score = np.inf
    best_params = None
    best_B = None
    all_results = []  # store ((lam1, lam2, rs), MSE)

    for lam1 in lambda1_list:
        for lam2 in lambda2_list:
            for rs in ramp_size_list:
                scores = []
                for train_index, val_index in kf.split(X):
                    X_train, X_val = X[train_index], X[val_index]
                    Y_train, Y_val = Y[train_index], Y[val_index]

                    B_candidate, _ = growl_fista(X_train, Y_train,
                                                 lambda_1=lam1,
                                                 lambda_2=lam2,
                                                 ramp_size=rs,
                                                 max_iter=100,
                                                 tol=1e-2,
                                                 check_type='solution_diff',
                                                 scale_objective=True,
                                                 verbose=False)
                            
                    # Predict on validation fold
                    Y_pred = X_val @ B_candidate # shape: (n_val, r)
                    
                    # Compute a metric, e.g. MSE or RMSE (we use MSE)
                    val_mse = np.mean((Y_val - Y_pred)**2)
                    scores.append(val_mse)

                avg_val_score = np.mean(scores)
                
                all_results.append(((lam1, lam2, rs), avg_val_score))

                print(f"(λ1={lam1:.2f}, λ2={lam2:.2f}, ramp_size={rs:.2f}) -> CV MSE = {avg_val_score:.6f}")

                if avg_val_score < best_score:
                    best_score = avg_val_score
                    best_params = (lam1, lam2, rs)
                    best_B = B_candidate.copy()

    return best_score, best_params, best_B, all_results


def grid_search_CV(X, Y, cross_validate_fn, lambda1_list_start, 
                   lambda2_list_start, ramp_size_list_start, n_stages=3, 
                   n_splits=5, random_state=42, min_improvement=1e-3, 
                   plot=True, verbose=True):
    """
    Conducts a staged parameter refinement by narrowing down the search space
    around the best parameters found in each stage. Stage 1 uses user-defined
    parameter grids (lambda1_list_start, lambda2_list_start, ramp_size_list_start).

    For each subsequent stage i = 2, ..., n_stages:
      - Measures how much CV MSE improved from the previous stage.
      - Dynamically chooses the factor (for lambda_1 and lambda_2) and the
        additive delta (for ramp_size) based on the improvement size.
      - If improvement is below 'min_improvement', we stop early.

    Optionally plots the CV performance vs. each parameter after the final stage
    (controlled by `plot`). Verbose printing is controlled by `verbose`.

    Parameters
    ----------
    X, Y : np.array
        Input and output data (already scaled) for the growl regression.
    cross_validate_fn : callable
        A function that performs cross-validation (e.g. `cross_validate_growl`).
    lambda1_list_start : array-like
        The grid of lambda_1 values to use in Stage 1.
    lambda2_list_start : array-like
        The grid of lambda_2 (lambda_2) values to use in Stage 1.
    ramp_size_list_start : array-like
        The grid of ramp_size values to use in Stage 1.
    n_stages : int, default=3
        The total number of stages (refinements) to perform (maximum, ignoring early stopping).
    n_splits : int, default=5
        Number of folds for KFold in cross-validation.
    random_state : int, default=42
        Random seed for reproducibility in cross-validation.
    plot : bool, default=True
        Whether to plot the CV performance vs. each parameter after the final stage.
    verbose : bool, default=True
        Whether to print progress and intermediate CV MSEs.
    min_improvement : float or None, default=1e-3
        If the relative improvement in MSE from the previous stage
        is below this value, we stop refining early. Set to None (or 0) to disable.

    Returns
    -------
    best_score_final : float
        The best MSE score found in the final (or last completed) stage.
    best_params_final : tuple
        The best (lambda1, lambda2, ramp_size) found in the final (or last completed) stage.
    best_B_final : np.array
        The coefficient matrix corresponding to the best parameters in the final stage.
    all_stages_results : list
        A list with results from all stages, e.g.:
            [ (best_score_1, best_params_1, best_B_1, all_results_1),
              (best_score_2, best_params_2, best_B_2, all_results_2),
              ...
              (best_score_n, best_params_n, best_B_n, all_results_n) ].
    """

    # --------------------------------------------------
    # 1) Helper: log-spaced grid around a center value
    # --------------------------------------------------
    def around_log_space(center, num, factor=2):
        """
        Returns a log-spaced grid of size 'num' approximately between
        [center/factor, center*factor].
        """
        low = max(center / factor, 1e-12)  # avoid zero
        high = max(center * factor, 1e-12)
        return np.logspace(np.log10(low), np.log10(high), num)

    # --------------------------------------------------
    # 2) Helper: ramp_size is typically a percentage, so we move ±delta
    # --------------------------------------------------
    def around_perc(center, delta=0.05):
        return [max(1e-6, center - delta), center, min(center + delta, 1)]

    # --------------------------------------------------
    # 3) Choose factor/delta based on improvement
    # --------------------------------------------------
    def choose_factor_delta(improvement):
        """
        Decide how large a factor/delta to use based on how much
        the MSE improved relative to the previous stage.

        improvement = (old_MSE - new_MSE) / old_MSE

        Returns
        -------
        (factor, delta)
        """
        if improvement > 0.3:
            # Large improvement -> keep factor & delta large
            return 3.0, 0.15
        elif improvement > 0.1:
            # Moderate improvement
            return 2.0, 0.10
        else:
            # Small improvement -> narrower refinement
            return 1.5, 0.05

    # We'll store the results of each stage
    all_stages_results = []

    # =========================
    # Stage 1: Use user-provided grids
    # =========================
    if verbose:
        print("===== Stage 1 =====")

    best_score_1, best_params_1, best_B_1, all_results_1 = cross_validate_fn(
        X, Y,
        lambda1_list_start,
        lambda2_list_start,
        ramp_size_list_start,
        n_splits=n_splits,
        random_state=random_state
    )

    if verbose:
        print(f"Best params (Stage 1) = {best_params_1}, CV MSE = {best_score_1:.4f}\n")

    all_stages_results.append((best_score_1, best_params_1, best_B_1, all_results_1))

    current_best_score = best_score_1
    current_best_params = best_params_1
    current_best_B = best_B_1

    # We'll define "last_score" to measure improvement from one stage to the next
    # Since Stage 1 has no previous stage, define last_score=best_score_1
    last_score = best_score_1

    # =========================
    # Stages 2, ..., n_stages
    # =========================
    for stage_i in range(2, n_stages + 1):
        lam1_prev, lam2_prev, rs_prev = current_best_params

        # Calculate improvement from last stage
        # If stage_i=2, we artificially set an 'improvement' as if we
        # made a big jump, so we can do broad exploration in stage 2.
        if stage_i == 2:
            improvement = 1.0  # big
        else:
            improvement = (last_score - current_best_score) / last_score if last_score > 0 else 1.0

        # Optionally, if improvement is extremely small => stop
        if (min_improvement is not None) and (improvement < min_improvement):
            if verbose:
                print(f"Stopping early at Stage {stage_i} due to small improvement ({improvement:.6f}).")
            break

        # Choose factor/delta for THIS stage based on last stage's improvement
        factor, delta = choose_factor_delta(improvement)

        # Build new grids around previous best
        lambda1_list = around_log_space(lam1_prev, factor=factor, num=len(lambda1_list_start))
        lambda2_list = around_log_space(lam2_prev, factor=factor, num=len(lambda2_list_start))
        ramp_size_list = around_perc(rs_prev, delta=delta)
        
        if stage_i > 2:
            print(f"Improvement from Stage {stage_i-2} to Stage {stage_i-1}: {improvement:.4f}\n")

        if verbose:
            print(f"===== Stage {stage_i} =====")
            print(f"Refining with factor={factor}, delta={delta}")
            print(f"lambda_1 grid: {np.round(lambda1_list, 4)}")
            print(f"lambda_2 grid: {np.round(lambda2_list, 4)}")
            print(f"ramp_size grid: {np.round(ramp_size_list, 4)}")
            print()
    
        # Run cross-validation for Stage i
        best_score_i, best_params_i, best_B_i, all_results_i = cross_validate_fn(
            X, Y,
            lambda1_list,
            lambda2_list,
            ramp_size_list,
            n_splits=n_splits,
            random_state=random_state
        )

        if verbose:
            print(f"Best params (Stage {stage_i}) = {best_params_i}, CV MSE = {best_score_i:.4f}\n")

        # Store results
        all_stages_results.append((best_score_i, best_params_i, best_B_i, all_results_i))

        # Update "current best" for the next iteration
        last_score = current_best_score
        current_best_score = best_score_i
        current_best_params = best_params_i
        current_best_B = best_B_i

    # After final stage or early stopping, we optionally do final plots
    best_score_final = current_best_score
    best_params_final = current_best_params
    best_B_final = current_best_B

    if plot:
        # Helper: 1D sweep for a single parameter while fixing others
        def sweep_one_param(X, Y, param_name, values, lam1_fixed, lam2_fixed, rs_fixed):
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            results = []
            for v in values:
                scores = []
                for train_index, val_index in kf.split(X):
                    X_train, X_val = X[train_index], X[val_index]
                    Y_train, Y_val = Y[train_index], Y[val_index]

                    if param_name == 'lambda_1':
                        B_candidate, _ = growl(
                            X_train, Y_train,
                            lambda_1=v,
                            lambda_2=lam2_fixed,
                            ramp_size=rs_fixed,
                            max_iter=100,
                            tol=1e-2,
                            check_type='solution_diff',
                            scale_objective=True,
                            verbose=False
                        )
                    elif param_name == 'lambda_2':
                        B_candidate, _ = growl(
                            X_train, Y_train,
                            lambda_1=lam1_fixed,
                            lambda_2=v,
                            ramp_size=rs_fixed,
                            max_iter=100,
                            tol=1e-2,
                            check_type='solution_diff',
                            scale_objective=True,
                            verbose=False
                        )
                    elif param_name == 'ramp_size':
                        B_candidate, _ = growl(
                            X_train, Y_train,
                            lambda_1=lam1_fixed,
                            lambda_2=lam2_fixed,
                            ramp_size=v,
                            max_iter=100,
                            tol=1e-2,
                            check_type='solution_diff',
                            scale_objective=True,
                            verbose=False
                        )
                    else:
                        raise ValueError("Unknown parameter name.")

                    Y_pred = X_val @ B_candidate
                    mse_val = np.mean((Y_val - Y_pred) ** 2)
                    scores.append(mse_val)
                results.append((v, np.mean(scores)))
            return results

        # Build small 1D grids around final best for plotting
        final_lam1, final_lam2, final_rs = best_params_final

        # Just a small factor/delta for the 1D sweeps:
        lambda1_sweep = around_log_space(final_lam1, factor=2, 
                                         num=len(lambda1_list_start))
        lambda2_sweep = around_log_space(final_lam2, factor=2, 
                                         num=len(lambda2_list_start))
        ramp_size_sweep = around_perc(final_rs, delta=0.05)

        # Plot vs. lambda_1
        results_lam1 = sweep_one_param(X, Y, 'lambda_1',
                                       lambda1_sweep,
                                       final_lam1, final_lam2, final_rs)
        plt.figure()
        x_vals = [r[0] for r in results_lam1]
        y_vals = [r[1] for r in results_lam1]
        plt.plot(x_vals, y_vals, marker='o')
        plt.xscale('log')
        plt.xlabel('lambda_1')
        plt.ylabel('CV MSE')
        plt.title(f'CV Performance vs. lambda_1\n'
                  f'(lambda_2={final_lam2}, ramp_size={final_rs})')
        plt.show()

        # Plot vs. lambda_2
        results_lam2 = sweep_one_param(X, Y, 'lambda_2',
                                       lambda2_sweep,
                                       final_lam1, final_lam2, final_rs)
        plt.figure()
        x_vals = [r[0] for r in results_lam2]
        y_vals = [r[1] for r in results_lam2]
        plt.plot(x_vals, y_vals, marker='o')
        plt.xscale('log')
        plt.xlabel('lambda_2')
        plt.ylabel('CV MSE')
        plt.title(f'CV Performance vs. lambda_2\n'
                  f'(lambda_1={final_lam1}, ramp_size={final_rs})')
        plt.show()

        # Plot vs. ramp_size
        results_rs = sweep_one_param(X, Y, 'ramp_size',
                                     ramp_size_sweep,
                                     final_lam1, final_lam2, final_rs)
        plt.figure()
        x_vals = [r[0] for r in results_rs]
        y_vals = [r[1] for r in results_rs]
        plt.plot(x_vals, y_vals, marker='o')
        plt.xlabel('ramp_size')
        plt.ylabel('CV MSE')
        plt.title(f'CV Performance vs. ramp_size\n'
                  f'(lambda_1={final_lam1}, lambda_2={final_lam2})')
        plt.show()

    if verbose:
        print("Done! Final best params:", best_params_final,
              "with CV MSE:", best_score_final)

    return best_score_final, best_params_final, best_B_final, all_stages_results

if __name__ == "__main__":
    
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    
    # 1) Prepare the data 
    obs_F = pd.read_csv("obs_F.csv", index_col=0)
    PCs = pd.read_csv("pred.csv", index_col=0)
    
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    
    obs_F_normalized = scaler_X.fit_transform(obs_F)
    PCs_normalized = scaler_Y.fit_transform(PCs)
    
    # 2) Small grids just to test the CV algorithm:
    lambda1_list_start = np.logspace(1, 2, 2) 
    lambda2_list_start = np.logspace(-2, -1, 2)
    ramp_size_list_start = [0.7, 0.9]
        
    # Run CV with the adaptive grid refinement approach
    best_score, best_params, best_B, stages_info = grid_search_CV(
        obs_F_normalized, PCs_normalized,
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
    
    print("Final best score:", best_score)
    print("Final best params:", best_params)
