function model_order = estimate_model_order(data, num_lags_max, regmode)
    % data: [nvars x nobs x ntrials]
    % num_lags_max: Maximum number of lags to consider
    % regmode: Regression mode ('OLS' or 'LWR')
    % Calculate information criteria (AIC, BIC) for model orders from pmin to num_lags_max
    pmin = 1; % Minimum model order
    [AIC, BIC] = tsdata_to_infocrit(data, num_lags_max, regmode); %tsdata_to_infocrit(data, pmin, num_lags_max, 'OLS');
    % Select the model order with minimum AIC
    [~, model_order_idx] = min(AIC);
    model_order = model_order_idx + pmin - 1; % Adjust for pmin
end