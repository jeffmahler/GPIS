% generate visualization for beta distribution

% flat distribution to start off
true_theta = [0.25, 0.75, 0.5];
num_arms = size(true_theta, 2);
alpha = ones(1, num_arms);
beta = ones(1, num_arms);

% sample values
pdf_points = 0:0.01:1;
num_samples = 10000;

sample_ind = 1;
colors = distinguishable_colors(num_arms);
num_pulls = zeros(num_arms, 1);
legend_labels = cell(2*num_arms, 1);

figure(1);
for i = 1:num_samples
    clf;
    max_p = 0;
    for k = 1:num_arms
        p = betapdf(pdf_points, alpha(k), beta(k));
        plot(pdf_points, p, 'Color', colors(k,:), 'LineWidth', 2);
        max_p = max(max(p), max_p);
        hold on;
    end
    
    
    for k = 1:num_arms
        legend_labels{k} = sprintf('Arm %d prior: %d samples', k, num_pulls(k));
        legend_labels{num_arms+k} = sprintf('True theta arm %d', k);
        plot([true_theta(k), true_theta(k)], [0,max_p+1], 'Color', colors(k,:),'LineWidth', 2);
    end
    
    xlim([0,1]);
    ylim([0,max_p+1]);
    title('Thompson sampling');
    legend(legend_labels{:}, 'Location', 'northwest');
    pause(0.001);
    
    % choose arm to pull with thompson
    best_arm_sample = 0;
    arm_pull = 0;
    for k = 1:num_arms
        arm_sample = betarnd(alpha(k), beta(k));
        if arm_sample > best_arm_sample
            best_arm_sample = arm_sample;
            arm_pull = k;
        end
    end
    
    % pull an arm using the true dist
    num_pulls(arm_pull) = num_pulls(arm_pull) + 1;
    unif_sample = rand();
    if unif_sample > true_theta(arm_pull)
        beta(arm_pull) = beta(arm_pull) + 1;   % update failures
    else
        alpha(arm_pull) = alpha(arm_pull) + 1; % update successes
    end
end

