function results = create_grasp_results_struct(grasps, meanQ, varQ, ...
    successes, satisfied, times, numGrasps)
%CREATE_GRASP_RESULTS_STRUCT just throw these things into a struct to save
%retyping

results = struct();
results.grasps = grasps;
results.meanQ = meanQ;
results.varQ = varQ;
results.successes = successes;
results.satisfied = satisfied;
results.times = times;
results.numGrasps = numGrasps;

end

