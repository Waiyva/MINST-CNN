function [W1,W3,W4] = DeltaMiniBatch(W1,W3,W4,X,D,N)
    alpha = 0.01;
    V1 = zeros(20,20,20);
    M = 5;
    
    for i = 1:(N/M)
        dW3sum = zeros(100,2000);
        dW4sum = zeros(10,100);
        for j = 1:M
            id = (i-1)*M+j;
            x = X(:,:,id);
            d = D(:,id);
            %滤波
            for epoch = 1:20
                V1(:,:,epoch) = filter2(W1(:,:,epoch), x, 'valid');
            end
            %ReLU
            Y1 = max(0,V1);
            %2×2平均池化
            Y2 = (Y1(1:2:end,1:2:end,:)+Y1(2:2:end,1:2:end,:)+Y1(1:2:end,2:2:end,:)+Y1(2:2:end,2:2:end,:))/4;
            %Reshape变为列向量
            y2 = reshape(Y2,[],1);
            v3 = W3*y2;
            %ReLU
            y3 = max(0,v3);
            v = W4*y3;
            %Softmax
            y = Softmax(v);

            %向前传播误差，并更新权重
            e = d - y;
            delta = e;
            e3 = W4'*delta;
            delta3 = (v3 > 0).*e3;
            e2 = W3'*delta3;
            dW4 = alpha*delta*y3';
            dW3 = alpha*delta3*y2';

            E2 = reshape(e2,size(Y2));
            E2_4 = E2/4;
            E1 = zeros(20,20,20);
            E1(1:2:end,1:2:end,:) = E2_4;
            E1(1:2:end,2:2:end,:) = E2_4;
            E1(2:2:end,1:2:end,:) = E2_4;
            E1(2:2:end,2:2:end,:) = E2_4;

            delta1 = (V1>0).*E1;
            dW1 = zeros(9,9,20);
            for epoch = 1:20
                dW1(:,:,epoch) = alpha*filter2(delta1(:,:,epoch),x,'valid');
            end
            W1 = W1 + dW1;
            dW4sum = dW4sum + dW4;
            dW3sum = dW3sum + dW3;
        end
        dW3avg = dW3sum / M;
        dW4avg = dW4sum / M;
        W3=W3+dW3avg;
        W4=W4+dW4avg;
    end   
    
end