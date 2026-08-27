function F = mvgc_analysis(data, model_order, alpha, regmode)
    % data: [nvars x nobs x ntrials]
    % model_order: Model order (number of lags)
    % alpha: Significance level
    % regmode: Regression mode ('OLS' or 'LWR')
    % Fit VAR model
    [A, SIG] = tsdata_to_var(data, model_order, regmode);
    assert(~isbad(A), 'VAR estimation failed');

    % Check VAR model stability
    % Construct the companion matrix
    [nvars, ~, ~] = size(data);
    Am = reshape(A, nvars, nvars * model_order);
    companion_matrix = [Am; eye(nvars * (model_order - 1)), zeros(nvars * (model_order - 1), nvars)];

    % Calculate eigenvalues
    eigenvalues = eig(companion_matrix);

    % Check if all eigenvalues are within the unit circle
    if any(abs(eigenvalues) >= 1)
        warning('VAR model is not stable. Eigenvalues outside the unit circle detected.');
        % Handle instability (e.g., reduce model order or regularization)
        % For now, we can attempt to reduce the model order
        while model_order > 1
            model_order = model_order - 1;
            [A, SIG] = tsdata_to_var(data, model_order, regmode);
            assert(~isbad(A), 'VAR estimation failed');

            % Reconstruct companion matrix and check stability again
            Am = reshape(A, nvars, nvars * model_order);
            companion_matrix = [Am; eye(nvars * (model_order - 1)), zeros(nvars * (model_order - 1), nvars)];
            eigenvalues = eig(companion_matrix);

            if all(abs(eigenvalues) < 1)
                disp(['Reduced model order to ', num2str(model_order), ' to achieve stability.']);
                break;
            end
        end

        if any(abs(eigenvalues) >= 1)
            error('VAR model is not stable even after reducing model order.');
        end
    end
    
    % Calculate autocovariance sequence
    [G, info] = var_to_autocov(A, SIG);

    % エラー情報の表示
    if info.error
        disp(['Error in var_to_autocov: ', info.errmsg]);
        error('VAR to autocovariance failed.');
    end
    
    % Calculate pairwise conditional Granger causality
    F = autocov_to_pwcgc(G);
    % assert(~isbad(F), 'Granger causality calculation failed');

    % Significance testing
    nvars = size(data, 1); % Total number of variables (neurons)
    nobs = size(data, 2);  % Number of observations (time points)
    ntrials = size(data, 3); % Number of trials
    N = nobs * ntrials;    % Total number of observations
    nx = 1; % Number of variables in X (from-variable)
    ny = 1; % Number of variables in Y (to-variable)
    nz = nvars - nx - ny; % Number of conditioning variables
    m = nvars * model_order; % Degrees of freedom for the full model
    tstat = 'F'; % Test statistic type

    % Compute p-values
    pval = zeros(nvars, nvars);
    for i = 1:nvars
        for j = 1:nvars
            if i ~= j
                F_ij = F(i, j);
                pval(i, j) = mvgc_pval(F_ij, model_order, m, N, nx, ny, nz, tstat);
            end
        end
    end

    % Apply significance mask
    sig = pval < alpha;
    F(~sig) = 0;
end

