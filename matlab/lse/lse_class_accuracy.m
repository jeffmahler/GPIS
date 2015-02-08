function class_results = ...
    lse_class_accuracy(true_above, true_below, pred_above, pred_below)
%LSE_CLASS_ACCURACY Summary of this function goes here
%   Detailed explanation goes here
class_results = struct();

class_results.fp = sum(pred_above & true_below);
class_results.tp = sum(pred_above & true_above);
class_results.fn = sum(pred_below & true_above);
class_results.tn = sum(pred_below & true_below);
class_results.ukn = sum(~pred_above & ~pred_below);

% takes unknowns into account
class_results.precision = class_results.tp / ...
    (class_results.tp + class_results.fp + class_results.ukn);
class_results.recall = class_results.tp / ...
    (class_results.tp + class_results.fn + class_results.ukn);
class_results.F1 = 2 * class_results.precision * class_results.recall / ...
    (class_results.precision + class_results.recall);

% do not take unknowns into account
class_results.std_precision = class_results.tp / ...
    (class_results.tp + class_results.fp);
class_results.std_recall = class_results.tp / ...
    (class_results.tp + class_results.fn);
class_results.std_F1 = 2 * class_results.std_precision * class_results.std_recall / ...
    (class_results.std_precision + class_results.std_recall);

% unknown rate (something like convergence rate)
class_results.ukn_rate = class_results.ukn / size(true_above, 1);

end

