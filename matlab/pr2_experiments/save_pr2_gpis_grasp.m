% saves a grasp from the experiments to pr2 format

outputDir = 'results/pr2/pfc';

% convert mean grasp
%meanGraspPr2 = convert_grasp_to_cb_frame(bestMeanGrasps{1}.bestGrasp', ...
%   transformResults, shapeNames{1}, outputDir);

% convert PFC grasp
pfcGraspPr2 = convert_grasp_to_cb_frame(bestPredGrasps{1}.expPGrasp.bestGrasp', ...
    transformResults, shapeNames{1}, outputDir);
