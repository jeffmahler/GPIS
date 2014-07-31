function [ contact_emps,norm_emps] = sample_loas(gpModel, loa, numSamples,grip_point)
%SAMPLE_SHAPE Summary of this function goes here
%   Detailed explanation goes here 
    numSamples = 1000; 
    COV = gp_cov(gpModel, loa, [], true);
    MEAN = gp_mean(gpModel, loa, true); 

    contact_emps= []; 
    norm_emps = []; 

    for i = 1:numSamples
        % make sure the sample is somewhat probable (almost everything will
        % evaluate to inf)
        sample = mvnrnd(MEAN, COV + 1e-12*eye(size(COV,1)));
        size(sample)
        loa_sample = sample(:,1:size(loa,1)); 
       
        idx = find(loa_sample <=0); 
        if(size(idx) ~= 0)
            contact_emps = [contact_emps; idx(1)];
            [marg_cov,marg_mean] = marg_normals(COV,MEAN,idx(1)); 
            norm_sample = mvnrnd(marg_mean, marg_cov + 1e-12*eye(size(marg_cov,1)));
            norm_emps = [norm_emps; norm_sample];
        end
        
        
        
    end
        
    
 

end

function [marg_cov,marg_mean] = marg_normals(cov_loa,mean_loa,i)

sz = size(mean_loa,1)/3;
dx = sz+i; 
dy = 2*sz+i;

marg_mean = [mean_loa(dx); mean_loa(dy)];

marg_cov = [cov_loa(dx,dx) cov_loa(dx,dy); 
            cov_loa(dx,dy) cov_loa(dy,dy)];

end

