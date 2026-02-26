% LeverPushPull.m
classdef LeverPushPull
    properties
        Ns = 2; % Number of states
        Na = 2; % Number of actions
        R;      % Reward matrix
        state;  % Current state
    end
    
    methods
        function obj = LeverPushPull()
            % Constructor
            obj.R = zeros(obj.Ns, obj.Na);
            % Pythonのインデックスは0から、MATLABは1からなので調整
            % R[0,0] = 2/3 -> R(1,1) = 2/3
            % R[1,1] = 2/3 -> R(2,2) = 2/3
            obj.R(1,1) = 2/3;
            obj.R(2,2) = 2/3;
        end
        
        function state = start(obj)
            % Start a new episode
            obj.state = randi(obj.Ns);
            state = obj.state;
        end
        
        function [reward, next_state] = step(obj, action)
            % Take a step
            % MATLABは1-based indexなので、stateとactionをそのまま使用
            reward = obj.R(obj.state, action);
            obj.state = randi(obj.Ns); % Next state is random
            next_state = obj.state;
        end
    end
end