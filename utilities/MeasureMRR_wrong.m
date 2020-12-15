function [ MRR ] = MeasureMRR( vec_label, vec_predict, Ks )
    %
    % A measure for nDCG @ k
    %
    K = max(Ks);
    
    [res, loc] = maxk(vec_predict', K);
    
    MRR = zeros(size(vec_predict, 1), length(Ks));
    for u = 1:size(vec_predict, 1)
        sort_indx_predict = loc(:, u);
        [sort_val_label, sort_indx_label] = sort(vec_label(u, :), 'descend');
        
        vecVal = zeros(length(sort_indx_label), 1);
        vecVal(sort_indx_label) = 1 ./ [1:length(sort_indx_label)];
        vecVal = vecVal .* (vec_label(u,:)>0)';
        
        for i = 1:length(Ks)
            k = Ks(i);
            MRR(u, i) = sum(vecVal(sort_indx_predict(1:k)));       
        end
    end
end