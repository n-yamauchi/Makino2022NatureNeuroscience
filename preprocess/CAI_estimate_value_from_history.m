% CAI_estimate_value_from_history.m
function [V_est, Q_est, A_est] = CAI_estimate_value_from_history(Trsa, num_states, num_actions, gamma)
    % この関数は、シミュレーション履歴(Trsa)から状態価値(V),行動価値(Q),
    % アドバンテージ(A)をモデルフリー（モンテカルロ法）で推定します。
    % 手法：初回訪問モンテカルロ法 (First-visit Monte Carlo)
    %
    % 出力:
    %   V_est: 各状態の推定価値 (Ns x 1)
    %   Q_est: 各状態・行動ペアの推定価値 (Ns x Na)
    %   A_est: 各状態・行動ペアの推定アドバンテージ (Ns x Na)

    % --- 初期化 ---
    V_returns_sum = zeros(num_states, 1);
    V_returns_count = zeros(num_states, 1);
    Q_returns_sum = zeros(num_states, num_actions);
    Q_returns_count = zeros(num_states, num_actions);
    
    T = size(Trsa, 1) - 1;
    rewards = Trsa(2:end, 2);  % 報酬シーケンス (t=1...T)
    states = Trsa(1:end-1, 3); % 状態シーケンス (t=0...T-1)
    actions = Trsa(1:end-1, 4);% 行動シーケンス (t=0...T-1)
    
    % --- 各タイムステップからの収益を計算 ---
    for t = 1:T
        s = states(t);
        a = actions(t);
        
        % このエピソードで、この状態sが初めて訪問されたかチェック
        is_first_visit_s = ~ismember(s, states(1:t-1));
        
        % このエピソードで、この状態-行動ペア(s,a)が初めてかチェック
        prev_pairs = [states(1:t-1), actions(1:t-1)];
        is_first_visit_sa = ~ismember([s, a], prev_pairs, 'rows');
        
        % 収益 G_t (tからエピソード終了までの割引報酬和) を計算
        % G_t は s, a 両方の計算で共通なので一度だけ計算
        if is_first_visit_s || is_first_visit_sa
            G = 0;
            for k = t:T
                G = G + gamma^(k-t) * rewards(k);
            end
        end

        % V(s) のための集計
        if is_first_visit_s
            V_returns_sum(s) = V_returns_sum(s) + G;
            V_returns_count(s) = V_returns_count(s) + 1;
        end
        
        % Q(s,a) のための集計
        if is_first_visit_sa
            Q_returns_sum(s, a) = Q_returns_sum(s, a) + G;
            Q_returns_count(s, a) = Q_returns_count(s, a) + 1;
        end
    end
    
    % --- 平均を計算して価値を推定 ---
    V_est = zeros(num_states, 1);
    Q_est = zeros(num_states, num_actions);
    
    % V(s) の計算 (ゼロ除算を回避)
    visited_s = V_returns_count > 0;
    V_est(visited_s) = V_returns_sum(visited_s) ./ V_returns_count(visited_s);
    
    % Q(s,a) の計算 (ゼロ除算を回避)
    visited_sa = Q_returns_count > 0;
    Q_est(visited_sa) = Q_returns_sum(visited_sa) ./ Q_returns_count(visited_sa);

    % --- アドバンテージ A(s,a) = Q(s,a) - V(s) を計算 ---
    % V_estを各列にブロードキャストして減算
    A_est = Q_est - V_est;
end