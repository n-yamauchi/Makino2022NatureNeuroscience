% CAI.m
classdef CAI
    properties
        Ns; Na; T; env_type;
        beta = 10;
        V; Q; Advantage;
        R; transition_matrix;
        mean_reward;
        step_count = 0;
        reward_schedule; % For LeverPushPullSwitch
        
        % Logging variables
        logged_action_value = {};
        logged_next_state_prediction = {};
        logged_reward_next = {};
        logged_state_value = {};
        logged_advantage = {};
        logged_action = {};
    end
    
    methods
        function obj = CAI(nstate, naction, tmax, envType)
            % Constructor
            obj.Ns = nstate;
            obj.Na = naction;
            obj.T = tmax;
            obj.env_type = envType;
            
            % Initialize tables
            obj.V = zeros(nstate, 1);
            obj.Q = zeros(nstate, naction);
            obj.Advantage = zeros(nstate, naction);
            
            % Define internal models based on env_type
            if strcmp(obj.env_type, "LeverPushPull")
                obj.R = zeros(obj.Ns, obj.Na);
                obj.R(1,1) = 2/3;
                obj.R(2,2) = 2/3;
                % NOTE: The agent's transition model is fixed to random values
                % This might differ from the environment's true dynamics.
                obj.transition_matrix = randi(obj.Ns, obj.Ns, obj.Na);
            elseif strcmp(obj.env_type, "LeverPushPullSwitch")
                R_LPP = zeros(obj.Ns, obj.Na); R_LPP(1,1) = 1.0; R_LPP(2,2) = 1/3;
                R2_LPP = zeros(obj.Ns, obj.Na); R2_LPP(1,1) = 1/3; R2_LPP(2,2) = 1.0;
                R3_LPP = zeros(obj.Ns, obj.Na); R3_LPP(1,1) = 2/3; R3_LPP(2,2) = 2/3;

                obj.reward_schedule = zeros(obj.T, 1);
                obj.reward_schedule(1:20) = 3;
                block_lengths = randi([35, 45], 1, 4);
                current_idx = 21;
                for i = 1:4
                    block_len = block_lengths(i);
                    end_idx = min(current_idx + block_len - 1, obj.T);
                    if mod(i, 2) == 1; reward_type = 1; else; reward_type = 2; end
                    obj.reward_schedule(current_idx:end_idx) = reward_type;
                    current_idx = end_idx + 1;
                    if current_idx > obj.T; break; end
                end
                if current_idx <= obj.T
                    obj.reward_schedule(current_idx:end) = 3;
                end
                obj.R = R3_LPP; % Initial Reward
                obj.transition_matrix = randi(obj.Ns, obj.Ns, obj.Na);
            end
            obj.mean_reward = mean(obj.R(:));
        end
        
        function action = start(obj, state)
            % Initial action
            q_adv = obj.Advantage(state, :);
            action = obj.boltzmann(q_adv);
            obj = obj.log_state(state);
        end

        function [obj, action] = step(obj, ~, state)
            % Update and select next action
            obj.step_count = obj.step_count + 1;

            % --- Update internal models if necessary ---
            if strcmp(obj.env_type, "LeverPushPullSwitch")
                current_trial = min(obj.step_count, obj.T);
                reward_type = obj.reward_schedule(current_trial);
                if reward_type == 1
                    obj.R(1,1) = 1.0; obj.R(1,2)=0; obj.R(2,1)=0; obj.R(2,2) = 1/3;
                elseif reward_type == 2
                    obj.R(1,1) = 1/3; obj.R(1,2)=0; obj.R(2,1)=0; obj.R(2,2) = 1.0;
                else
                    obj.R(1,1) = 2/3; obj.R(1,2)=0; obj.R(2,1)=0; obj.R(2,2) = 2/3;
                end
                obj.mean_reward = mean(obj.R(:));
            end
            
            % --- Inference/Control Update ---
            next_states = obj.transition_matrix(state, :);
            rewards = obj.R(state, :);
            
            % Q-value update
            obj.Q(state, :) = rewards - obj.mean_reward + obj.V(next_states)';
            
            % V-value update (using log-sum-exp for stability)
            log_q = obj.Q(state, :);
            log_w = obj.Advantage(state, :);
            obj.V(state) = logsumexp(log_q + log_w) - logsumexp(log_w);

            % Advantage update
            adv_raw = obj.Q(state, :) - obj.V(state);
            obj.Advantage(state, :) = min(max(adv_raw, -100), 100);
            
            % Action selection
            action = obj.boltzmann(obj.Advantage(state, :));
            
            % Logging
            obj = obj.log_state(state, next_states, rewards, action);
        end
        
        function action_idx = boltzmann(obj, q)
            % Boltzmann (softmax) action selection
            p = exp(obj.beta * q);
            p(p < 1e-10) = 1e-10;
            p_sum = sum(p);
            if p_sum <= 1e-10
                p = ones(1, length(q)) / length(q);
            else
                p = p / p_sum;
            end
            % Choose action based on probability
            action_idx = randsample(length(p), 1, true, p);
        end

        function obj = log_state(obj, state, next_states, rewards, action)
            if nargin < 3
                next_states = zeros(1, obj.Na);
                rewards = zeros(1, obj.Na);
                action = NaN;
            end
            obj.logged_action_value{end+1} = obj.Q(state, :);
            obj.logged_next_state_prediction{end+1} = next_states;
            obj.logged_reward_next{end+1} = rewards;
            obj.logged_state_value{end+1} = obj.V(state);
            obj.logged_advantage{end+1} = obj.Advantage(state, :);
            obj.logged_action{end+1} = action;
        end

        function obj = update_from_observation(obj, state)
            % このメソッドは、与えられた状態(state)に基づいて
            % 内部の価値テーブル(Q, V, Advantage)のみを更新します。
            % 行動選択は行いません。
            
            obj.step_count = obj.step_count + 1;

            % --- 環境タイプに応じた内部報酬モデルの更新 ---
            if strcmp(obj.env_type, "LeverPushPullSwitch")
                current_trial = min(obj.step_count, obj.T);
                reward_type = obj.reward_schedule(current_trial);
                if reward_type == 1
                    obj.R(1,1) = 1.0; obj.R(1,2)=0; obj.R(2,1)=0; obj.R(2,2) = 1/3;
                elseif reward_type == 2
                    obj.R(1,1) = 1/3; obj.R(1,2)=0; obj.R(2,1)=0; obj.R(2,2) = 1.0;
                else
                    obj.R(1,1) = 2/3; obj.R(1,2)=0; obj.R(2,1)=0; obj.R(2,2) = 2/3;
                end
                obj.mean_reward = mean(obj.R(:));
            end
            
            % --- 推論/制御による価値更新 ---
            next_states = obj.transition_matrix(state, :);
            rewards = obj.R(state, :);
            
            % Q-value update
            obj.Q(state, :) = rewards - obj.mean_reward + obj.V(next_states)';
            
            % V-value update (using log-sum-exp for stability)
            log_q = obj.Q(state, :);
            log_w = obj.Advantage(state, :);
            obj.V(state) = logsumexp(log_q + log_w) - logsumexp(log_w);

            % Advantage update
            adv_raw = obj.Q(state, :) - obj.V(state);
            obj.Advantage(state, :) = min(max(adv_raw, -100), 100);
        end
    end
end

function lse = logsumexp(x)
    % Numerically stable log-sum-exp
    c = max(x);
    lse = c + log(sum(exp(x - c)));
end