% read in caging logs and plot

data_dir = 'data/caging/square';
files = dir(data_dir);
num_files = size(files,1);

colors = distinguishable_colors(num_files);
labels = {};
pct_cages = [];
cage_rate_arr = [];

figure(1);
for i = 1:num_files
    % make sure actual file
    if ~files(i).isdir
        filename = sprintf('%s/%s', data_dir, files(i).name);
        f = fopen(filename, 'r');
        file_data = fscanf(f, '%f\n');
        
        pct_cage = file_data(1);
        cage_rates = file_data(2:end);
        pct_cages = [pct_cages, pct_cage];
        cage_rate_arr = [cage_rate_arr, cage_rates];
    end
end

index = 1;
num_cage_exps = size(pct_cages,2);
[~, sorted_ind] = sort(pct_cages);
for i = 1:num_cage_exps
    ind = sorted_ind(i);
    
    plot(cage_rate_arr(:,ind), 'Color', colors(i, :), 'LineWidth', 2);
    hold on;
    labels{index} = sprintf('%.03f pct caged', pct_cages(ind));
    index = index + 1;
end

legend(labels{:}, 'Location', 'EastOutside');
xlabel('Samples', 'FontSize', 15);
ylabel('Caged Rate', 'FontSize', 15);
title('Cage Rate vs Num Samples', 'FontSize', 15);