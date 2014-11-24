function [] = sdf_surface(tsdfGrid, scale)
%SDF_SURFACE Summary of this function goes here
%   Detailed explanation goes here

if nargin < 2
    scale = 2;
end

tsdfGridBig = high_res_tsdf(tsdfGrid, scale);

tsdfColors = zeros(size(tsdfGridBig,1), size(tsdfGridBig,2), 3);
tsdfColors(:,:,1) = ones(size(tsdfGridBig));
tsdfColors(:,:,2) = ones(size(tsdfGridBig));
surf(tsdfGridBig, tsdfColors);%, 'LineStyle', 'none');
xlabel('X  coord (px)');
ylabel('Y  coord (px)');
zlabel('Signed Distance');

% set(gca, 'XTick', []);
% set(gca, 'YTick', []);
hold on;

zeroCrossing = zeros(size(tsdfGridBig));
%colormap([0,0,1]);
zcColors = zeros(size(tsdfGridBig,1), size(tsdfGridBig,2), 3);
zcColors(:,:,3) = ones(size(tsdfGridBig));
surf(zeroCrossing, zcColors);
%colorbar

tsdfThresh = tsdfGridBig > 0;
SE = strel('square', 3);
I_d = imdilate(tsdfThresh, SE);

% create border masks
insideMaskOrig = (tsdfThresh == 0);
outsideMaskDi = (I_d == 1);
tsdfSurface = double(~(outsideMaskDi & insideMaskOrig));
[interiorI, interiorJ] = find(insideMaskOrig == 1);
%plot3(interiorJ, interiorI, -2*ones(size(interiorI,1),1), ...
%    'LineWidth', 1, 'Color', [1,0,0]);
end

