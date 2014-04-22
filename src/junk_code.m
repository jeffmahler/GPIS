
%% Create a gpis representation from the samples using a random active set
% numIters = 10;
% thresh = 0.075;
% gpModel = create_gpis(points, tsdf, numIters);
% [testPoints, testTsdf, testVars, surfacePoints, surfaceTsdf, surfaceVars] = ...
%     predict_2d_grid(gpModel, gridDim, thresh);


%testOther = min(1, testImage + 0.05*ones(4*gridDim, 4*gridDim)); % brighten
%testEq = histeq(testOther, 255);
%testEq = max(0, testEq - 0.1*ones(4*gridDim, 4*gridDim)); % darken

%
alpha = 0.4;
combined = alpha*testImage + (1-alpha)*testVarImage;
