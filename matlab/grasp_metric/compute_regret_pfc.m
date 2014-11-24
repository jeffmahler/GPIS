function [ regret ] = compute_regret_pfc(grasp_number)
%COMPUTE_REGRET Summary of this function goes here
%   Detailed explanation goes here


    load('marker_bandit_values_pfc.mat'); 
    [v,best_grasp] = max(Value(:,3)); 
    
    v_picked = Value(grasp_number,3); 
    
    regret = v - v_picked; 
end