function [] = interpolate_sdfs(tsdf1, tsdf2, gridDim,  ...
    sdfScale, imScale, delta)
%INTERPOLATE_SDFS Summary of this function goes here
%   Detailed explanation goes here
if nargin < 6
    delta = 0.1;
end

% plot initial sdfs
figure(1);
subplot(1,2,1);
sdf_surface(tsdf1, sdfScale);
subplot(1,2,2);
sdf_surface(tsdf2, sdfScale);

interpWeights = 0:delta:1;

% linear interpolation
for i = 1:size(interpWeights,2)
    t = interpWeights(i);
    interpTsdf = (1-t)*tsdf1 + t*tsdf2;
    figure(2);
    clf;
    sdf_surface(interpTsdf, sdfScale);
    view([135, 75]);
%     figure(3);
%     clf;
%     tsdfGrid = reshape(interpTsdf, [gridDim, gridDim]);
%     imagesc(high_res_tsdf(tsdfGrid, imScale));
    pause(1);
end

end

