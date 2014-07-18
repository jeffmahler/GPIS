function errors = evaluate_errors(predicted, true, K)

    errors = struct();
    diff = predicted - true;
    errors.modelSize = K;
    errors.rawError = diff;
    errors.absError = abs(diff);
    errors.meanError = mean(errors.absError,1);
    errors.medianError = median(errors.absError,1);
    errors.rmsError = sqrt(mean(errors.absError.^2,1));
    errors.stdError = std(errors.absError,1);

end

