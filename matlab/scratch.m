dim = shapeParams.gridDim;
gripScale = gripScales{1};
gripWidth = gripScale * experimentConfig.objScale * dim;
plateWidth = gripWidth * experimentConfig.plateScale;
plateWidth = uint16(round(plateWidth));

predGrid = experimentResults.constructionResults.predGrid;
surfaceImage = experimentResults.constructionResults.surfaceImage;
newSurfaceImage = experimentResults.constructionResults.newSurfaceImage;

% create struct for nominal shape
nominalShape = struct();
nominalShape.tsdf = shapeParams.fullTsdf;
nominalShape.normals = shapeParams.fullNormals;
nominalShape.points = shapeParams.all_points;
nominalShape.noise = zeros(size(nominalShape.tsdf,1), 1);
nominalShape.gridDim = shapeParams.gridDim;
nominalShape.surfaceThresh = shapeParams.surfaceThresh;
nominalShape.com = shapeParams.com;

newSurfaceImage = reshape(nominalShape.tsdf, 25, 25);
newSurfaceImage = imresize(newSurfaceImage, 2.0);

h = figure(12);
subplot(1,2,1);
visualize_grasp(bestMeanGrasps{1}.bestGrasp', nominalShape, newSurfaceImage, scale, ...
    experimentConfig.arrowLength, plateWidth, gripWidth);
title('Best Grasp for Mean Shape', 'FontSize', 10);
xlabel(sprintf('P(FC) = %.03f', bestMeanGrasps{1}.expP), 'FontSize', 10);
subplot(1,2,2);
visualize_grasp(bestPredGrasps{1}.expPGrasp.bestGrasp', nominalShape, newSurfaceImage, scale, ...
    experimentConfig.arrowLength, plateWidth, gripWidth);
title('Best Grasp for GPIS Using P(FC)', 'FontSize', 10);
xlabel(sprintf('P(FC) = %.03f', bestPredGrasps{1}.expPGrasp.P), 'FontSize', 10);

%print(h, '-depsc', sprintf('%s/%s_comp2_p.eps', outputDir, filename));
