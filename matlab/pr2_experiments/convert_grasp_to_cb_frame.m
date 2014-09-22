function graspCbFrame = ...
    convert_grasp_to_cb_frame(graspGridFrame, transformParams, ...
        graspName, outputDir)
%CONVERT_GRASP_TO_CB_FRAME Convert a grasp computed in the grid frame to
% a grasp in the chessboard frame, for deployment on the PR2
%  
% graspGridFrame - 4x1 vector (first two elements are grasp point 1, second are
%          grasp point 2

% read parameters
T_cb_grid = transformParams.T_cb_grid;
pixTo3D = transformParams.pixTo3D; % convert image pix to 3D coords
gpScale = transformParams.gpScale; % image scale to gp scale

% get grasp endpoints and centroid
d = 2;
g1 = graspGridFrame(1:d,1);
g2 = graspGridFrame(d+1:2*d,1);
graspCenter = (g1 + g2) / 2;

% transform points to cb frame (in 0-Z plane)
g1Cb = inv(T_cb_grid) * [pixTo3D * (1.0 / gpScale) * g1; 0; 1];
g2Cb = inv(T_cb_grid) * [pixTo3D * (1.0 / gpScale) * g2; 0; 1];
graspCenterCb = inv(T_cb_grid) * [pixTo3D * (1.0 / gpScale) * graspCenter; 0; 1];

% don't use homog coords for direction
graspDirCb = (g2Cb(1:3) - g1Cb(1:3));
graspDirCb = graspDirCb / norm(graspDirCb);

% save grasp center, direction, and endpoints
graspCenterFilename = sprintf('%s/%s_center.csv', outputDir, graspName);
graspDirFilename = sprintf('%s/%s_dir.csv', outputDir, graspName);
graspEndpointsFilename = sprintf('%s/%s_endpoints.csv', outputDir, graspName);

csvwrite(graspCenterFilename, graspCenterCb);
csvwrite(graspDirFilename, graspDirCb);
csvwrite(graspEndpointsFilename, [g1Cb; g2Cb]);

graspCbFrame = struct();
graspCbFrame.center = graspCenterCb;
graspCbFrame.dir = graspDirCb;
graspCbFrame.g1 = g1Cb;
graspCbFrame.g2 = g2Cb;

end

