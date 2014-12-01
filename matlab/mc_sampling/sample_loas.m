function [ contact_emps, norm_emps] = sample_loas(gpModel, loa, numSamples, pose_samples)
%SAMPLE_SHAPE Summary of this function goes here
%   Detailed explanation goes here 
  
    % default to identity poses
    use_pose = true;
    if nargin < 4
       use_pose = false;
    end

    contact_emps= []; 
    norm_emps = [];
    grid_center = mean(gpModel.training_x)';
    grid_center = repmat(grid_center, 1, size(loa, 1));
    
    if ~use_pose
        COV = gp_cov(gpModel, loa, [], true);
        MEAN = gp_mean(gpModel, loa, true);
        shape_samples = mvnrnd(MEAN, COV + 1e-12*eye(size(COV,1)), numSamples);
    end
            
    for i = 1:numSamples
        %fprintf('Sample %d\n', i);
        % make sure the sample is somewhat probable (almost everything will
        % evaluate to inf)
        %s
        
        if use_pose
            pose_sample = pose_samples{i};

            % transform loa
            loa_transformed = [loa' - grid_center; ...
                ones(1, size(loa, 1))];
            loa_transformed = inv(pose_sample) * loa_transformed;
            loa_transformed = loa_transformed(1:2,:) + grid_center;

            startTime = tic;
            COV = gp_cov(gpModel, loa_transformed', [], true);
            MEAN = gp_mean(gpModel, loa_transformed', true);
            endTime = toc(startTime);
            %fprintf('Mat time %f\n', endTime);

            startTime = tic;
            shape_sample = mvnrnd(MEAN, COV + 1e-12*eye(size(COV,1)), 1);
            endTime = toc(startTime);
            %fprintf('Sample time %f\n', endTime);
        else
           shape_sample = shape_samples(i,:); 
        end
        
        loa_sample = shape_sample(:,1:size(loa,1)); 
        norm_sample = reshape(shape_sample(:,size(loa,1)+1:end),size(loa,1),2);
       
        idx = find(loa_sample <= 0.05); 
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