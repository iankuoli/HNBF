function [ nDCG ] = MeasureNDCG( vec_label, vec_predict, Ks )
    %
    % A measure for nDCG @ k
    %
    K = max(Ks);
    
    [res, loc] = maxk(vec_predict', K);
    
    nDCG = zeros(size(vec_predict, 1), length(Ks));
    for u = 1:size(vec_predict, 1)
        [sort_val_predict, sort_indx_predict] = sort(res(:,u), 'descend');
        [sort_val_label, sort_indx_label] = sort(vec_label(u, :), 'descend');
        
        vecRel = vec_label(u, loc(sort_indx_predict, u));
        
        for i = 1:length(Ks)
            k = Ks(i);
            DCG = sum(vecRel(1:k) ./ log2((1:k)+1));
            IDCG = sum(sort_val_label(1:k) ./ log2((1:k)+1));
            nDCG(u, i) = DCG / IDCG;
        end
    end

end