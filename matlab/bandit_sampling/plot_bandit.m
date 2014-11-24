load('regret_marker_pfc_mc.mat');

regret_mc = regret; 
regret_mc(end) = regret(end-1); 

load('regret_marker_pfc.mat');

regret_bandit = regret; 
offset = 0; 
regret_bandit_big = zeros(size(regret_mc))%+regret(end); 

regret_bandit_big(1:size(regret_bandit,1)) = regret_bandit; 


load('regret_marker_pfc_sf.mat');

regret_sf = regret; 
offset = 0; 
regret_bandit_sf = zeros(size(regret_mc))%+regret(end); 

regret_bandit_sf(1:size(regret_bandit_sf,1)) = regret_sf; 

figure; 
hold on
font = 18;
plot(regret_bandit_big-offset,'r-','LineWidth',3);
plot(regret_bandit_sf-offset,'b-','LineWidth',3);
title('Sampling Methods','FontSize',font);
xlabel('Samples','FontSize',font);
ylabel('Simple Regret','FontSize',font);
[hleg1, hobj1] = legend('Thompson','Science Fair','FontSize',font);
textobj = findobj(hobj1, 'type', 'text');
set(textobj, 'Interpreter', 'latex', 'fontsize', 20);
axis([1000 size(regret_bandit_big,1) 0 0.2]); 
hold off