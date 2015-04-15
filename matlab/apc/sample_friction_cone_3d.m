function v_samples = ...
    sample_friction_cone_3d(cone_support, num_samples, dir_prior)

if nargin < 3
    dir_prior = 0.5;
end

num_faces = size(cone_support, 2);
v_samples = zeros(3, num_samples);
for j = 1:num_samples
    lambdas = gamrnd(dir_prior * ones(num_faces, 1), dir_prior);
    lambdas = lambdas / sum(lambdas);
    v_sample = repmat(lambdas', [3,1]) .* cone_support;
    v_samples(:,j) = sum(v_sample, 2);
end

end

