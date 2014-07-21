function [ allTsdf,allNorms] = sample_shape( gpModel,allPoints)
%SAMPLE_SHAPE Summary of this function goes here
%   Detailed explanation goes here

num_points = size(allPoints,1); 
gridDim = 25; 
scale =2 ;

COV = gp_cov(gpModel,allPoints, [], true);

MEAN = gp_mean(gpModel,allPoints,true); 

sample = mvnrnd(MEAN,COV);

allTsdf = sample(1,1:num_points)'; 
allNorms = reshape(sample(1,num_points+1:end),num_points,2); 


% numTest = size(allPoints, 1);
% testColors = repmat(ones(numTest,1) - abs(allTsdf) / max(abs(allTsdf)) .* ones(numTest,1), 1, 3);
% 
% testImage = reshape(testColors(:,1), gridDim, gridDim); 
% testImage = imresize(testImage, scale*size(testImage));
% 
% testImageDarkened = testImage;
% max(0, testImage - 0.3*ones(scale*gridDim, scale*gridDim)); % darken
%  
% figure;
% imshow(testImageDarkened);

end

