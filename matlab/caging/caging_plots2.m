% Plot the new caging results

data_dir = 'data/caging/alpha_vs_size';
files = dir(data_dir);
num_files = size(files,1);

colors = distinguishable_colors(num_files-2);
pct_cages = [];
cage_rate_arr = [];

labels = cell(num_files-2, 1);

figure(1);
k = 1;
for i = 1:num_files
    if ~files(i).isdir
        filename = sprintf('%s/%s', data_dir, files(i).name);
        f = fopen(filename, 'r');
        file_data = fscanf(f, '%f %f %f\n');
        
        s = file_data(1:3:end);
        a = file_data(3:3:end);
        
        if k == 1
            sizes = zeros(size(s,1), num_files-2);
            alphas = zeros(size(s,1), num_files-2);
            samples = zeros(num_files-2,1);
        end
        sizes(:,k) = s;
        alphas(:,k) = a;
        samples(k) = sscanf(files(i).name, 'tri_scaling_%d_samples.txt');
        
        k = k+1;
    end
end

[~, sorted_ind] = sort(samples);
num_samples = size(samples,1);
for i = 1:num_samples
    k = sorted_ind(i);
    labels{i} = sprintf('%d samples', samples(k));    
    plot(sizes(:,k), alphas(:,k), 'LineWidth', 4, 'Color', colors(i,:));
    hold on;
end

xlim([min(min(sizes)), max(max(sizes))]);
ylim([min(min(alphas)), max(max(alphas))]);
xlabel('Triangle Size', 'FontSize', 15);
ylabel('Cage Alpha', 'FontSize', 15);
title('Cage Alpha vs Size', 'FontSize', 15);
legend(labels{:}, 'Location', 'Best');


%% plot the sample variance vs size

data_dir = 'data/caging/alpha_vs_samples';
files = dir(data_dir);
num_files = size(files,1);

colors = distinguishable_colors(num_files-2);
pct_cages = [];
cage_rate_arr = [];
labels = cell(num_files-2, 1);

figure(2);
k = 1;
for i = 1:num_files
    if ~files(i).isdir
        filename = sprintf('%s/%s', data_dir, files(i).name);
        f = fopen(filename, 'r');
        file_data = fscanf(f, '%f %f %f\n');
        
        a = file_data(3:3:end);
        if k == 1
            samp = size(a,1);
            alphas = zeros(samp, num_files-2);
            samples = zeros(num_files-2,1);
            avg_alphas = zeros(num_files-2,1);
            std_alphas = zeros(num_files-2,1);
        end
        
        alphas(:,k) = a;
        avg_alphas(k) = mean(a);
        std_alphas(k) = std(a);
        samples(k) = sscanf(files(i).name, '%d_samples.txt');
        k = k+1;
    end
end

[~, sorted_ind] = sort(samples);
num_samples = size(samples,1);

for i = 1:num_samples
    k = sorted_ind(i);
    labels{i} = sprintf('%d samples', samples(k));    
    scatter(repmat(log(samples(k)), [samp, 1]), alphas(:,k), 50, 'LineWidth', 2, ...
        'MarkerFaceColor', colors(2,:), 'MarkerEdgeColor', colors(2,:));
    hold on;
end
errorbar(log(samples(sorted_ind)), avg_alphas(sorted_ind), std_alphas(sorted_ind),...
        'LineWidth', 4, ...
        'Color', colors(1,:));  
hold on;

xlim(log([min(samples)-100, max(samples)+100]));
ylim([min(min(alphas)), max(max(alphas))]);
xlabel('Log Num Samples', 'FontSize', 15);
ylabel('Cage Alpha', 'FontSize', 15);
title('Cage Alpha vs Num Samples', 'FontSize', 15);
%legend(labels{:}, 'Location', 'Best');

%% timing results

data_dir = 'data/caging/timing';
files = dir(data_dir);
num_files = size(files,1);

colors = distinguishable_colors(num_files-2);
pct_cages = [];
cage_rate_arr = [];
labels = cell(num_files-2, 1);

figure(3);
k = 1;
for i = 1:num_files
    if ~files(i).isdir
        filename = sprintf('%s/%s', data_dir, files(i).name);
        f = fopen(filename, 'r');
        file_data = fscanf(f, '%f %f %f\n');
        
        s = file_data(1:4:end);
        a = file_data(3:4:end);
        t = file_data(4:4:end);
        if k == 1
            samp = size(a,1);
            siz = size(s,1);
            sizes = zeros(siz, num_files-2);
            alphas = zeros(siz, num_files-2);
            times = zeros(siz, num_files-2);
            samples = zeros(num_files-2, 1);
        end
        
        sizes(:,k) = s;
        alphas(:,k) = a;
        times(:,k) = t;
        samples(k) = sscanf(files(i).name, 'timing_%d_sampling.txt');
        k = k+1;
    end
end

[~, sorted_ind] = sort(samples);
num_samples = size(samples,1);
for i = 1:num_samples
    k = sorted_ind(i);
    labels{i} = sprintf('%d samples', samples(k));    
    
    [unique_sizes, ia, ic] = unique(sizes(:,k));
    num_sizes = size(unique_sizes, 1);
    avg_times = zeros(num_sizes, 1);
    std_times = zeros(num_sizes, 1);
    for j = 1:num_sizes
        avg_times(j) = mean(times(ic == j,k));
        std_times(j) = std(times(ic == j,k));
    end
    
    errorbar(unique_sizes, avg_times, std_times,...
        'LineWidth', 4, 'Color', colors(k,:)); 
    %plot(unique_sizes, times(:,k), 'LineWidth', 4, 'Color', colors(k,:));
    hold on;
end

xlim([min(min(sizes)), max(max(sizes))]);
ylim([min(min(times)), max(max(times))]);
xlabel('Triangle Side Length', 'FontSize', 15);
ylabel('Runtime (sec)', 'FontSize', 15);
title('Cage Alpha Computation Time vs Size and Samples', 'FontSize', 15);
legend(labels{:}, 'Location', 'Best');