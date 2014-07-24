function [ allTsdf,allNorms] = sample_shape( gpModel,allPoints)
%SAMPLE_SHAPE Summary of this function goes here
%   Detailed explanation goes here

num_points = size(allPoints,1); 
gridDim = 25; 
scale =2 ;

COV = gp_cov(gpModel,allPoints, [], true);

MEAN = gp_mean(gpModel,allPoints,true); 

pdf_thresh = 0.1;
sample_pdf = 0;

% make sure the sample is somewhat probable (almost everything will
% evaluate to inf)
while ~isinf(sample_pdf) && sample_pdf < pdf_thresh
    sample = mvnrnd(MEAN,COV);
    sample_pdf = mvnpdf(sample, MEAN', COV + 1e-14*eye(size(COV,1)));
end

allTsdf = sample(1,1:num_points)'; 
allNorms = reshape(sample(1,num_points+1:end),num_points,2); 

end

