function graspSamples = sample_grasps(loa, graspSigma, N)
%SAMPLES_GRASPS generate N samples of grasps

graspSamples = cell(1,N);

for i = 1:N
    graspSamples{i} = sample_grasp(loa, graspSigma);
end

end

