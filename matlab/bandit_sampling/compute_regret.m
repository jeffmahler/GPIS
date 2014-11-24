function [ regret ] = compute_regret(grasp_number)
%COMPUTE_REGRET Summary of this function goes here
%   Detailed explanation goes here


    load('marker_bandit_values_pfc.mat'); 
    [v,best_grasp] = max(Value(:,1)); 
    
    v_picked = Value(grasp_number,1); 
    
    regret = v - v_picked; 
end

