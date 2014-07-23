function [gpModel, shapeParams, shapeSamples, constructionResults] = ...
    load_experiment_object(filename, dataDir)

% load GPIS and shape
shapeName = sprintf('%s/%s.mat', dataDir, filename);
gpisName = sprintf('%s/%s_gpis.mat', dataDir, filename);
samplesName = sprintf('%s/%s_samples.mat', dataDir, filename);
constructName = sprintf('%s/%s_construction.mat', dataDir, filename);
varName = sprintf('%s/%s_variance_params.mat', dataDir, filename);

load(shapeName, 'shapeParams');
load(gpisName, 'gpModel');
load(samplesName, 'shapeSamples');
load(constructName, 'constructionResults');
load(varName, 'varParams');
end

