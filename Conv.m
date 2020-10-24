function V1 = Conv(X,W1)
    V1 = zeros(20,20,20);
    %ÂË²¨
    for epoch = 1:20
        V1(:,:,epoch) = filter2(W1(:,:,epoch), X, 'valid');
    end
end