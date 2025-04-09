function result = randnumtable(weighted_matrix, n_context)
    % Randsample values based on a weighted matrix and 3D n_context
    % Inputs:
    %   - weighted_matrix: [m x n x p] サンプリング確率の重み行列
    %   - n_context: [m x n x p] 各位置でのサンプリング回数
    % Outputs:
    %   - result: サンプリング結果のインデックスまたは値 (3D配列)
    
    % 行列の次元を取得
    [rows, cols, depth] = size(weighted_matrix);
    
    % 正規化 (各スライスで確率を列方向に合計1にする)
    probabilities = weighted_matrix ./ sum(weighted_matrix, 1, 'omitnan');
    
    % サンプリング結果を格納する配列
    result = zeros(rows, cols, depth);
    
    % 各スライスごとにランダムサンプリングを実行
    for d = 1:depth
        for col = 1:cols
            % 現在の列の確率分布を取得
            prob_dist = probabilities(:, col, d);

            % サンプリング数
            num_samples = n_context(:, col, d);

            % 各行ごとにサンプリング
            for r = 1:rows
                % サンプリング（置き換えあり）
                result(r, col, d) = randsample(rows, num_samples(r), true, prob_dist);
            end
        end
    end
end




