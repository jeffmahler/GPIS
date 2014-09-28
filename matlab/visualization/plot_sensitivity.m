% Plot sensitivity

offset = 1.5; % num seconds to sample shape
length = 4; % arrow length
plateScale = 0.075;
objScale = 0.95;
dim = 25;

shapeNames = {'plane', 'can_opener', 'tape'};
gripScales = {1.2, 0.4, 0.6};
hashNums = {736205, 555647, 412912};%{555647, 163603, 24685, 736205, 241629, 658943, 412912, 34654};
dataDir = 'results/optimization/icra';
hyperDir = 'results/hyper_tuning';
%meanCompDir = 'results/mean_vs_predicted_exp/icra_long';
numShapes = size(shapeNames, 2);
numHyper = size(hyperResults,2);

for i = 1:numShapes
    shapeName = shapeNames{i};
    hashNum = hashNums{i};
    gripScale = gripScales{i};
    optResultName = sprintf('%s/experiment_%s_%d.mat', dataDir, shapeName, hashNum);
    load(optResultName);
    
    gripWidth = gripScale * objScale * dim;
    plateWidth = gripWidth * plateScale;
    plateWidth = uint16(round(plateWidth));
    if plateWidth == 0
        plateWidth = 1;
    end
    
    predGrid = experimentResults.constructionResults.predGrid;
    surfaceImage = experimentResults.constructionResults.newSurfaceImage;
    
    h = figure(42+i);
    hold on;
    for j = 1:numHyper
        optResults = hyperResults{1,j};
        optGrasp = zeros(4,1);
        optP = 0;
        scale = 4;
        
        validOptInd = optResults.success > 0 & optResults.satisfied > 0;
        if size(validOptInd,2) > 0
            optInd = find(optResults.opt_vals == ...
                min(optResults.opt_vals(validOptInd)));
        
            optGrasp = optResults.grasps(:,optInd);
            optP = optResults.p_fc(optInd);
        end
        
        lambda = optResults.cfg.nu;
        
        subplot(1,numHyper,j);
        visualize_grasp(optGrasp, predGrid, surfaceImage, ...
            scale, length, plateWidth, gripWidth);
        title(sprintf('Lambda = %f', lambda), 'FontSize', 12, 'FontWeight', 'bold');
        xlabel(sprintf('P(FC) = %.03f', ...
            optResults.p_fc(optInd)), ...
            'FontSize', 10); 
        hold on;
    end
    print(h, '-depsc', sprintf('%s/%s_sensitivity.eps', hyperDir, shapeName));

    
end