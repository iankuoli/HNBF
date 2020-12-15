function [ usr_idx, itm_idx, usr_idx_len, itm_idx_len ] = sampleData_userwise(usr_batch_size)

    %% Global parameter declaration
    
    global M
    global N
  
    global matX_train
    
    
    %% Sample data
    
    if usr_batch_size == M
        usr_idx = 1:M;
        itm_idx = 1:N;
        usr_idx(sum(matX_train(usr_idx,:),2)==0) = [];
        itm_idx(sum(matX_train(:,itm_idx),1)==0) = [];
    else
        if usr_batch_size == size(matX_train,1)
            usr_idx = 1:size(matX_train,1);
        else
            usr_idx = randsample(size(matX_train,1), usr_batch_size);
            usr_idx(sum(matX_train(usr_idx,:),2)==0) = [];
        end

        itm_idx = find(sum(matX_train(usr_idx, :))>0);
    end
    
    usr_idx_len = length(usr_idx);
    itm_idx_len = length(itm_idx);
end