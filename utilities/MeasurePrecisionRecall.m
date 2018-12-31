function [ precision, recall ] = MeasurePrecisionRecall( vec_label, vec_predict, Ks )
    %
    % A measure for precision and recall @ k
    %
    K = max(Ks);
    
    [res, loc] = maxk(vec_predict', K);
    
    [usr_id, topK_rank, item_id] = find(loc');
    
    precision = zeros(size(vec_predict, 1), length(Ks));
    recall = zeros(size(vec_predict, 1), length(Ks));
    for i = 1:length(Ks)
        win_size = size(vec_predict,1)*Ks(i);
        
        % dim(accurate_mask) = usr_size * itm_size
        accurate_mask = sparse(usr_id(1:win_size), item_id(1:win_size), ones(length(item_id(1:win_size)),1), size(vec_predict, 1), size(vec_predict, 2));
        accurate_mask = (accurate_mask .* vec_label) > 0;
        precision(:, i) = full(sum(accurate_mask, 2)) / Ks(i);
        recall(:, i) = full(sum(accurate_mask, 2) ./ sum(vec_label > 0, 2));
    end

end