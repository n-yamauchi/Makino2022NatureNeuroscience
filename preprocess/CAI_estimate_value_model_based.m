% CAI_estimate_value_model_based.m
function [V_hist, Q_hist, A_hist, agent] = CAI_estimate_value_model_based(Trsa, agent)
    % この関数は、与えられた時系列データ(Trsa)とCAIエージェントを用いて、
    % モデルベースで状態価値(V), 行動価値(Q), アドバンテージ(A)の変遷を
    % 1ステップずつ推定します。
    %
    % 出力:
    %   V_hist: 各タイムステップにおける全状態の価値 (T+1 x Ns)
    %   Q_hist: 各タイムステップにおける全状態・行動の価値 (T+1 x Ns x Na)
    %   A_hist: 各タイムステップにおける全状態・行動のアドバンテージ (T+1 x Ns x Na)
    %   agent:  全ての履歴を処理した後のエージェント

    tmax = size(Trsa, 1) - 1;
    ns = agent.Ns;
    na = agent.Na;
    
    % 価値の履歴を保存する配列を初期化
    V_hist = zeros(tmax + 1, ns);
    Q_hist = zeros(tmax + 1, ns, na);
    A_hist = zeros(tmax + 1, ns, na);
    
    % t=0 の初期価値を保存
    V_hist(1, :) = agent.V';
    Q_hist(1, :, :) = agent.Q;
    A_hist(1, :, :) = agent.Advantage;
    
    % 履歴データを1ステップずつ処理
    for t = 1:tmax
        state_to_update = Trsa(t, 3);
        
        % エージェントの価値テーブルを更新
        agent = agent.update_from_observation(state_to_update);
        
        % 更新後の全価値を記録
        V_hist(t+1, :) = agent.V';
        Q_hist(t+1, :, :) = agent.Q;
        A_hist(t+1, :, :) = agent.Advantage;
    end
end