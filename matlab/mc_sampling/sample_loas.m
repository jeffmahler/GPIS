function [ contact_emps, norm_emps] = sample_loas(gpModel, loa, numSamples, pose_samples)
%SAMPLE_SHAPE Summary of this function goes here
%   Detailed explanation goes here 
  
    % default to identity poses
    if nargin < 4
       pose_samples = cell(1, numSamples);
       for i = 1:numSamples
          pose_samples{i} = eye(4);
       end
    end

    COV = gp_cov(gpModel, loa, [], true);
    MEAN = gp_mean(gpModel, loa, true);
    
    contact_emps= []; 
    norm_emps = []; 
    shape_samples = mvnrnd(MEAN, COV + 1e-12*eye(size(COV,1)), numSamples);
    
    for i = 1:numSamples
        % make sure the sample is somewhat probable (almost everything will
        % evaluate to inf)
        shape_sample = shape_samples(i,:);
        pose_sample = pose_samples{i};
%         
%         theta = acos(pose_sample(1,1));
% %        t = pose(1:2,3);
%         shape_sample = imrotate(shape_sample, theta);
%         shape_sample = imtranslate(shape_sample, t);
%         
        loa_sample = shape_sample(:,1:size(loa,1)); 
        norm_sample = reshape(shape_sample(:,size(loa,1)+1:end),size(loa,1),2);
       
        idx = find(loa_sample <=0.05); 
        if(size(idx) ~= 0)
            contact_emps = [contact_emps; idx(1)];
            %[marg_cov,marg_mean] = marg_normals(COV,MEAN,idx(1)); 
           
            norm_emps = [norm_emps; norm_sample(idx(1),:)/norm(norm_sample(idx(1),:))];
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