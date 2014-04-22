function [gpModel, activePoints, testIndices, testTsdf, testVars] = ...
    select_active_set( method, points, tsdf, K, numIters, h, varScale, eps)

if strcmp(method, 'Random') == 1
    [gpModel, activePoints, testIndices, testTsdf, testVars] = ...
        select_active_set_random(points, tsdf, K, numIters);
elseif strcmp(method, 'Subsample') == 1
    [gpModel, activePoints, testIndices, testTsdf, testVars] = ...
        select_active_set_subsample(points, tsdf, K, numIters);
elseif strcmp(method, 'Entropy') == 1
    [gpModel, activePoints, testIndices, testTsdf, testVars] = ...
        select_active_set_entropy(points, tsdf, K, numIters);
elseif strcmp(method, 'LevelSet') == 1
    [gpModel, activePoints, testIndices, testTsdf, testVars] = ...
        select_active_set_straddle(points, tsdf, K, numIters, h, varScale, eps);
else 
    disp('Method not recognized');
    return;
end

end

