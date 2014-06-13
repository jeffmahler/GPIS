function errors = evaluate_errors(predicted, true, K)

    errors = struct();
    diff = predicted - true;
    errors.modelSize = K;
    errors.rawError = diff;
    errors.absError = abs(diff);
    errors.meanError = mean(errors.absError);
    errors.medianError = median(errors.absError);
    errors.rmsError = sqrt(mean(errors.absError.^2));
    errors.stdError = std(errors.absError);

end

