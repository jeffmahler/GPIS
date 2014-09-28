% Random things I like to do at the end of gpis_2d but they're not
% necessary

offset = 1.5; % num seconds to sample shape
length = 4; % arrow length
plateScale = 0.075;
objScale = 0.95;

shapeNames = {'can_opener', 'deodorant', 'marker', 'plane', 'squirt_bottle', 'stapler', 'tape', 'water'};
gripScales = {0.45, 0.6, 0.8, 1.2, 0.4, 0.6, 0.75, 0.4};
hashNums = {555647, 163603, 24685, 736205, 241629, 658943, 412912, 34654};
dataDir = 'results/optimization/icra';
outputDir = 'results/optimization';
meanCompDir = 'results/mean_vs_predicted_exp/icra_long';
numShapes = size(shapeNames, 2);

%temp
nomQ = [0.041; 0.010; -0.007; 0.092; 0.002; 0.038; 0.109; 0.126];

for i = 1:numShapes
    shapeName = shapeNames{i};
    hashNum = hashNums{i};
    gripScale = gripScales{i};
    optResultName = sprintf('%s/experiment_%s_%d.mat', dataDir, shapeName, hashNum);
    load(optResultName);
    % now its in experimentResults
    
    gripWidth = gripScale * objScale * dim;
    plateWidth = gripWidth * plateScale;
    plateWidth = uint16(round(plateWidth));
    if plateWidth == 0
        plateWidth = 1;
    end
    
    % load the best grasps on the nominal, mean, and GPIS shapes
    nomGraspFilename = sprintf('%s/%s_best_grasp_nom.mat', meanCompDir, shapeName);
    meanGraspFilename = sprintf('%s/%s_best_grasp_mean.mat', meanCompDir, shapeName);
    ucGraspFilename = sprintf('%s/%s_best_grasp_uncertainty.mat', meanCompDir, shapeName);

    load(nomGraspFilename);  % named bestNom
    load(meanGraspFilename); % named bestMean
    load(ucGraspFilename);   % named bestSampling
    
    % load mean
    meanGrasp = bestMean.bestGrasp;
    
    % load sampling
    samplingGrasp = bestSampling.expPGrasp.bestGrasp;
    
    % load opt grasp
    optResults = experimentResults.ucFcOptGraspResults;
    numOpt = size(optResults.grasps, 1);
    optGrasp = optResults.grasps(optResults.bestIndex, :);
    
    predGrid = experimentResults.constructionResults.predGrid;
    surfaceImage = experimentResults.constructionResults.newSurfaceImage;
    
    h = figure(39);
    subplot(1,3,1);
    visualize_grasp(meanGrasp', predGrid, surfaceImage, ...
        scale, length, plateWidth, gripWidth);
    title({'Best Grasp Ranked by', 'Q(F) on Mean Shape'}, 'FontSize', 12, 'FontWeight', 'bold');
    xlabel(sprintf('Q(N) = %.03f\n P(FC) = %.03f', ...
        bestMean.nomQ, bestMean.expP), ...
        'FontSize', 10);
%     xlabh = get(gca, 'XLabel');
%     set(xlabh,'Position',get(xlabh,'Position') + [0 1 0]);
    
    subplot(1,3,2);
    visualize_grasp(bestSampling.expPGrasp.bestGrasp', predGrid, surfaceImage, ...
        scale, length, plateWidth, gripWidth);
    title({'Best Grasp Ranked by', 'P(FC) on GPIS'}, 'FontSize', 12, 'FontWeight', 'bold');
    xlabel(sprintf('Q(N) = %.03f\n P(FC) = %.03f', ...
        bestSampling.expPGrasp.nomQ, ...
        bestSampling.expPGrasp.P), ...
        'FontSize', 10);    

    subplot(1,3,3);
    visualize_grasp(optGrasp', predGrid, surfaceImage, ...
        scale, length, plateWidth, gripWidth);
    title({'Grasp Selected by', 'Our Algorithm'}, 'FontSize', 12, 'FontWeight', 'bold');
    xlabel(sprintf('Q(N) = %.03f\n P(FC) = %.03f', ...
        nomQ(i), optResults.pFc(optResults.bestIndex)), ...
        'FontSize', 10); 

    print(h, '-depsc', sprintf('%s/%s_comp_mean_pfc_revised.eps', outputDir, shapeName));

    
end