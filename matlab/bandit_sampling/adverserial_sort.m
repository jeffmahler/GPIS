function [ grasp_samples ] = adverserial_sort( Value,grasp_samples )
%ADVERSERIAL_SORT Summary of this function goes here
%   Detailed explanation goes here

num_grasp = size(Value,1); 
prcent = 0.5; 

%bottom grasps
[sV,SI] = sort(Value(:,3)); 

for t=1:num_grasp*prcent
    Qs = sort(grasp_samples{SI(t)}.Q_samps,'descend'); 
    grasp_samples{SI(t)}.Q_samps = Qs; 
end

%top grasps 
[sV,SI] = sort(Value(:,3),'descend'); 

for t=1:num_grasp*prcent
    Qs = sort(grasp_samples{SI(t)}.Q_samps); 
    grasp_samples{SI(t)}.Q_samps = Qs; 
end



end

