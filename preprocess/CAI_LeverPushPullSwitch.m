% LeverPushPullSwitch.m
classdef LeverPushPullSwitch
    properties
        Ns = 2;
        Na = 2;
        T = 200;
        R_LPP;
        R2_LPP;
        R3_LPP;
        reward_schedule;
        reward_models;
        state;
        step_count = 0;
    end
    
    methods
        function obj = LeverPushPullSwitch(time_steps)
            % Constructor
            if nargin > 0
                obj.T = time_steps;
            end
            
            % Define reward models (1-based indexing)
            obj.R_LPP = zeros(obj.Ns, obj.Na); obj.R_LPP(1,1) = 1.0; obj.R_LPP(2,2) = 1/3;
            obj.R2_LPP = zeros(obj.Ns, obj.Na); obj.R2_LPP(1,1) = 1/3; obj.R2_LPP(2,2) = 1.0;
            obj.R3_LPP = zeros(obj.Ns, obj.Na); obj.R3_LPP(1,1) = 2/3; obj.R3_LPP(2,2) = 2/3;
            
            % Store models in a struct for easy access
            obj.reward_models.R1 = obj.R_LPP;
            obj.reward_models.R2 = obj.R2_LPP;
            obj.reward_models.R3 = obj.R3_LPP;

            % Pre-calculate reward schedule
            obj.reward_schedule = zeros(obj.T, 1);
            obj.reward_schedule(1:20) = 3;
            block_lengths = randi([35, 45], 1, 4);
            current_idx = 21;
            for i = 1:4
                block_len = block_lengths(i);
                end_idx = min(current_idx + block_len - 1, obj.T);
                if mod(i, 2) == 1
                    reward_type = 1;
                else
                    reward_type = 2;
                end
                obj.reward_schedule(current_idx:end_idx) = reward_type;
                current_idx = end_idx + 1;
                if current_idx > obj.T; break; end
            end
            if current_idx <= obj.T
                obj.reward_schedule(current_idx:end) = 3;
            end
        end
        
        function state = start(obj)
            % Start a new episode
            obj.state = randi(obj.Ns);
            obj.step_count = 0;
            state = obj.state;
        end
        
        function [reward, next_state] = step(obj, action)
            % Take a step
            obj.step_count = obj.step_count + 1;
            current_trial = min(obj.step_count, obj.T);
            
            % Select current reward model based on schedule
            reward_type = obj.reward_schedule(current_trial);
            switch reward_type
                case 1
                    current_reward_model = obj.reward_models.R1;
                case 2
                    current_reward_model = obj.reward_models.R2;
                otherwise
                    current_reward_model = obj.reward_models.R3;
            end
            
            reward = current_reward_model(obj.state, action);
            obj.state = randi(obj.Ns); % Next state is random
            next_state = obj.state;
        end
    end
end