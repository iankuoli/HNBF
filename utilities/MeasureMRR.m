function [ MRR ] = MeasureMRR( vec_label, vec_predict, Ks )
    %
    % A measure MRR
    %
    K = max(Ks);
    MRR = zeros(size(vec_predict, 1), 1);

    [res, loc] = max(vec_label');
    for u = 1:size(vec_predict, 1)
    	MRR(u, 1) = 1 / (sum((vec_predict(u,:) - vec_predict(u, loc(u)))>0) + 1);
    end
end