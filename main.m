%训练代码
clear;
%clc;
load('MNISTData.mat');
%计时开始
tic;
N = length(D_Train);
%初始化
W1 = randn(9,9,20);
W3 = (2*rand(100,2000)-1)/20;
W4 = (2*rand(10,100)-1)/10;

X = X_Train;
D = D_Train;
[W1,W3,W4] = DeltaMiniBatch(W1,W3,W4,X,D,N);


%测试代码
N = length(D_Test);
d_comp = zeros(1,N);
for k = 1:N
    X = X_Test(:,:,k);
    V1 = Conv(X,W1);%自定义函数（不旋转，直接滤波）
    Y1 = ReLU(V1);
    Y2 = Pool(Y1);%自定义函数，2x2平均池化操作
    
    y2 = reshape(Y2,[],1);
    v3 = W3*y2;
    y3 = ReLU(v3);
    v = W4*y3;
    y = Softmax(v);
    [~,i] = max(y);%找到y向量中的最大元素，i为其位置索引
    d_comp(k) = i;%保存CNN的计算值（识别出的数字）
end

[~,d_true] = max(D_Test);%将单热编码变回相应的数字，存入d_true（1xN维向量）
acc = sum(d_comp==d_true);%统计正确识别的总数
fprintf('Accuracy is %f\n', acc/N);%输出正确率
%计时结束
toc;



