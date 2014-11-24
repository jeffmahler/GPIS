% saves a grasp from the experiments to pr2 format

outputDir = 'results/pr2/mean';
% convert mean grasp
meanGraspPr2 = convert_grasp_to_cb_frame(bestMeanGrasps{1}.bestGrasp', ...
  transformResults, shapeNames{1}, outputDir);

%%
outputDir = 'results/pr2/pfc';
% convert PFC grasp
pfcGraspPr2 = convert_grasp_to_cb_frame(bestPredGrasps{1}.expPGrasp.bestGrasp', ...
    transformResults, shapeNames{1}, outputDir);
%%
% convert uc opt
outputDir = 'results/pr2/uc_pfc';
bestIndex = experimentResults.ucFcOptGraspResults.bestIndex;
pfcGraspPr2 = convert_grasp_to_cb_frame( ...
    experimentResults.ucFcOptGraspResults.grasps(bestIndex,:)', ...
    transformResults, shapeNames{1}, outputDir);