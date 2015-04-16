% Plot the caging results for object data

trials_per_object = 20;
data_dir = 'data/caging/objects';

filename = 'caging_sample_output.txt';
data = csvread(sprintf('%s/%s', data_dir, filename));
data = repair_object_ids(data, trials_per_object);

filename2 = 'caging_sample_output2.txt';
data2 = csvread(sprintf('%s/%s', data_dir, filename2));
data2 = repair_object_ids(data2, trials_per_object);

% get relevant numbers
num_data_points = size(data, 1);
num_objects = num_data_points / trials_per_object;
colors = distinguishable_colors(num_objects);

%% gather data per object
sample_vals = unique(data(:,2));
num_samples = size(sample_vals, 1);

obj_alphas = cell(num_objects,num_samples);
for i = 1:num_objects
    for j = 1:num_samples
        obj_alphas{i, j} = [];
    end
end

obj_sample_times = cell(num_objects, num_samples);
for i = 1:num_objects
    for j = 1:num_samples
        obj_sample_times{i, j} = [];
    end
end

obj_alpha_times = cell(num_objects, num_samples);
for i = 1:num_objects
    for j = 1:num_samples
        obj_alpha_times{i, j} = [];
    end
end

obj_times = cell(num_objects, num_samples);
for i = 1:num_objects
    for j = 1:num_samples
        obj_times{i, j} = [];
    end
end

obj_collr = cell(num_objects, num_samples);
for i = 1:num_objects
    for j = 1:num_samples
        obj_collr{i, j} = [];
    end
end

% collect data points
for i = 1:num_data_points
    obj_id = data(i,1)+1;
    samples = data(i,2);
    alpha = data(i,3);
    collr = data(i,4);
    sample_time = data(i,5);
    alpha_time = data(i,6);
    total_time = data(i,7);
    
    sample_index = find(samples == sample_vals);
    sample_index = sample_index(1);
    
    obj_alphas{obj_id, sample_index} = ...
        [obj_alphas{obj_id, sample_index}, alpha];
    
    obj_sample_times{obj_id, sample_index} = ...
        [obj_sample_times{obj_id, sample_index}, sample_time];
    
    obj_alpha_times{obj_id, sample_index} = ...
        [obj_alpha_times{obj_id, sample_index}, alpha_time];
    
    obj_times{obj_id, sample_index} = ...
        [obj_times{obj_id, sample_index}, total_time];
    
    obj_collr{obj_id, sample_index} = ...
        [obj_collr{obj_id, sample_index}, collr];
end

%% plot alphas
rng(100);
num_compare = 2;
subsample = 1;
obj_mean_alphas = zeros(num_objects, num_samples);
obj_med_alphas = zeros(num_objects, num_samples);
obj_std_alphas = zeros(num_objects, num_samples);
rand_ind = datasample(1:num_objects, num_compare, 'Replace', false);
rand_ind = 1:subsample:num_objects;
effective_num_objects = size(rand_ind, 2);
labels = cell(effective_num_objects, 1);
close all

figure(1);
clf;
for k = 1:effective_num_objects
   i = rand_ind(k);
   for j = 1:num_samples
       obj_mean_alphas(i,j) = mean(obj_alphas{i,j});
       obj_med_alphas(i,j) = median(obj_alphas{i,j});
       obj_std_alphas(i,j) = std(obj_alphas{i,j});
   end
   labels{k} = sprintf('Object %d', i-1);
   %errorbar(sample_vals, obj_mean_alphas(i,:),  obj_std_alphas(i,:), ...
   %    'Color', colors(i,:), 'LineWidth', 4);
   plot(sample_vals, obj_mean_alphas(i,:), ...%  obj_std_alphas(i,:), ...
       'Color', colors(i,:), 'LineWidth', 4);
   hold on;
%    plot(sample_vals, obj_med_alphas(i,:), ...
%        'Color', colors(i,:), 'LineWidth', 4, 'LineStyle', ':');
end
xlabel('Num Samples', 'FontSize', 15);
ylabel('Cage Alpha', 'FontSize', 15);
title('Cage Alphas vs Num Samples', 'FontSize', 15);
legend(labels{:}, 'Location', 'Best');

%% plot times
rng(100);
subsample = 10;
effective_num_objects = int16(num_objects / subsample);

obj_mean_sample_times = zeros(num_objects, num_samples);
obj_med_sample_times = zeros(num_objects, num_samples);
obj_std_sample_times = zeros(num_objects, num_samples);

obj_mean_alpha_times = zeros(num_objects, num_samples);
obj_med_alpha_times = zeros(num_objects, num_samples);
obj_std_alpha_times = zeros(num_objects, num_samples);

obj_mean_times = zeros(num_objects, num_samples);
obj_med_times = zeros(num_objects, num_samples);
obj_std_times = zeros(num_objects, num_samples);

labels = cell(effective_num_objects, 1);

figure(2);
clf;
for i = 1:subsample:num_objects
   %i = rand_ind(k);
   for j = 1:num_samples
       obj_mean_sample_times(i,j) = mean(obj_sample_times{i,j});
       obj_med_sample_times(i,j) = median(obj_sample_times{i,j});
       obj_std_sample_times(i,j) = std(obj_sample_times{i,j});
       
       obj_mean_alpha_times(i,j) = mean(obj_alpha_times{i,j});
       obj_med_alpha_times(i,j) = median(obj_alpha_times{i,j});
       obj_std_alpha_times(i,j) = std(obj_alpha_times{i,j});
       
       obj_mean_times(i,j) = mean(obj_times{i,j});
       obj_med_times(i,j) = median(obj_times{i,j});
       obj_std_times(i,j) = std(obj_times{i,j});
   end
   labels{int16(i/subsample)+1} = sprintf('Object %d', i-1);
   errorbar(sample_vals, obj_mean_times(i,:),  obj_std_times(i,:), ...
       'Color', colors(i,:), 'LineWidth', 4);
   hold on;
%    plot(sample_vals, obj_med_alphas(i,:), ...
%        'Color', colors(i,:), 'LineWidth', 4, 'LineStyle', ':');
end
xlabel('Num Samples', 'FontSize', 15);
ylabel('Runtime (sec)', 'FontSize', 15);
title('Runtime vs Num Samples', 'FontSize', 15);
legend(labels{:}, 'Location', 'EastOutside');

%% time percentages
all_mean_sample_times = mean(obj_mean_sample_times(obj_mean_sample_times(:,1) > 0, :), 1);
all_mean_alpha_times = mean(obj_mean_alpha_times(obj_mean_alpha_times(:,1) > 0, :), 1);
all_mean_times = mean(obj_mean_times(obj_mean_times(:,1) > 0, :), 1);
all_times_diff = all_mean_times ...
    - all_mean_sample_times - all_mean_alpha_times; 

figure(10);
area([all_mean_sample_times', all_mean_alpha_times', all_times_diff']);
legend('Sample Time', 'Alpha Shape Construction Time', ...
    'Alpha Search  (+ Small Overhead)', 'Location', 'Best');
xlabel('Num Samples', 'FontSize', 15);
ylabel('Runtime (sec)', 'FontSize', 15);
title('Average Runtime Broken Down by Process', 'FontSize', 15);

%% plot collision ratio
rng(100);
subsample = 10;
effective_num_objects = int16(num_objects / subsample);
obj_mean_collr = zeros(num_objects, num_samples);
obj_med_collr = zeros(num_objects, num_samples);
obj_std_collr = zeros(num_objects, num_samples);
obj_eff_samples = zeros(num_objects, num_samples);
labels = cell(effective_num_objects, 1);

figure(3);
clf;
for i = 1:subsample:num_objects
   %i = rand_ind(k);
   for j = 1:num_samples
       obj_mean_collr(i,j) = mean(obj_collr{i,j});
       obj_med_collr(i,j) = median(obj_collr{i,j});
       obj_std_collr(i,j) = std(obj_collr{i,j});
       obj_eff_samples(i,j) = obj_mean_collr(i,j) * sample_vals(j);
   end
   labels{int16(i/subsample)+1} = sprintf('Object %d', i-1);
   errorbar(sample_vals, obj_mean_collr(i,:),  obj_std_collr(i,:), ...
       'Color', colors(i,:), 'LineWidth', 4);
   hold on;
%    plot(sample_vals, obj_med_alphas(i,:), ...
%        'Color', colors(i,:), 'LineWidth', 4, 'LineStyle', ':');
end
xlabel('Num Samples', 'FontSize', 15);
ylabel('% Samples in Collision', 'FontSize', 15);
title('Sampling Collision Ratio vs Num Samples', 'FontSize', 15);
legend(labels{:}, 'Location', 'EastOutside');

obj_mean_eff_samples = mean(obj_eff_samples(obj_eff_samples(:,1) > 0, :), 1);
