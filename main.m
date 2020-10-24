%ѵ������
clear;
%clc;
load('MNISTData.mat');
%��ʱ��ʼ
tic;
N = length(D_Train);
%��ʼ��
W1 = randn(9,9,20);
W3 = (2*rand(100,2000)-1)/20;
W4 = (2*rand(10,100)-1)/10;

X = X_Train;
D = D_Train;
[W1,W3,W4] = DeltaMiniBatch(W1,W3,W4,X,D,N);


%���Դ���
N = length(D_Test);
d_comp = zeros(1,N);
for k = 1:N
    X = X_Test(:,:,k);
    V1 = Conv(X,W1);%�Զ��庯��������ת��ֱ���˲���
    Y1 = ReLU(V1);
    Y2 = Pool(Y1);%�Զ��庯����2x2ƽ���ػ�����
    
    y2 = reshape(Y2,[],1);
    v3 = W3*y2;
    y3 = ReLU(v3);
    v = W4*y3;
    y = Softmax(v);
    [~,i] = max(y);%�ҵ�y�����е����Ԫ�أ�iΪ��λ������
    d_comp(k) = i;%����CNN�ļ���ֵ��ʶ��������֣�
end

[~,d_true] = max(D_Test);%�����ȱ�������Ӧ�����֣�����d_true��1xNά������
acc = sum(d_comp==d_true);%ͳ����ȷʶ�������
fprintf('Accuracy is %f\n', acc/N);%�����ȷ��
%��ʱ����
toc;



