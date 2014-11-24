% Script to analyze the results of a hyperparameter search 

%load('results/hyper_tuning/results1/hyper_results.mat');

%% Search for the numbers with the highest mean and var Q

numShapes = size(hyperResults, 1);
numCombinations = size(hyperResults, 2);

bestQResults = cell(1, numShapes);
bestPResults = cell(1, numShapes);
bestQIndices = cell(1, numShapes);
bestPIndices = cell(1, numShapes);
avgOptTimes = zeros(numShapes, numCombinations);

for i = 1:numShapes
    bestQResults{i} = [];
    bestPResults{i} = [];
    bestQIndices{i} = [];
    bestPIndices{i} = [];
    
    % first pass - find the max mean Q and mean P
    maxMnQ = -realmax;
    maxMnP = -realmax;
    
    for j = 1:numCombinations
        % compute mean
        combMnQ = mean(hyperResults{i,j}.mn_q);
        combMnP = mean(hyperResults{i,j}.p_fc);
        avgOptTimes(i,j) = mean(hyperResults{i,j}.opt_times);
        
        if combMnQ > maxMnQ
           maxMnQ = combMnQ; 
        end
        if combMnP > maxMnP
           maxMnP = combMnP; 
        end
    end
    
    % second pass - get the combinations that achieve the maximum
    for j = 1:numCombinations
        % compute means
        combMnQ = mean(hyperResults{i,j}.mn_q);
        combMnP = mean(hyperResults{i,j}.p_fc);
       
        if abs(combMnQ - maxMnQ) < 1e-5
           bestQResults{i} = [bestQResults{i} hyperResults{i,j}];
           bestQIndices{i} = [bestQIndices{i} j];
        end
        if abs(combMnP - maxMnP) < 1e-5
           bestPResults{i} = [bestPResults{i} hyperResults{i,j}]; 
           bestPIndices{i} = [bestPIndices{i} j];
        end
    end
end
