function [surfaceTestIndices, surfaceTestPredTsdf, surfaceTestPredVars] = ...
    predict_2d_surface(truePoints, trueTsdf, gpModel, testIndices)


% get the test surface points
surfaceIndices = find(trueTsdf < 1);
surfaceTestIndices = intersect(testIndices, surfaceIndices);

surfaceTestPoints = truePoints(surfaceTestIndices,:);

[surfaceTestPredTsdf, surfaceTestPredVars] = gp(gpModel.hyp, @infExact, gpModel.meanFunc, ...
    gpModel.covFunc, gpModel.likFunc, gpModel.training_x, gpModel.training_y, surfaceTestPoints);

end

