clear;
clc;
close all;

%--------------------------Initializing----------------------------------------%
X=[0:0.01:1; 0:0.01:1];

Xin = X(:, 1:70);
D_Learn = (Xin(1,:).^ 2 + Xin(2,:) .^ 2) .* humps(Xin(1,:));
D_Learn = D_Learn / max(D_Learn(:));

X_valid = X(:, 71:90);
D_valid = (X_valid(1,:).^ 2 + X_valid(2,:) .^ 2) .* humps(X_valid(1,:));
D_valid = D_valid / max(D_valid(:));

X_test = X(:, 91:101);
D_test = (X_test(1,:).^ 2 + X_test(2,:) .^ 2) .* humps(X_test(1,:));
D_test = D_test/max(D_test(:));
%--------------------------Feedforward and Backpropagation----------------------------------------%

eta=0.01;
epsilon=0.1;
%--------------------------16 neuron in 1st hidden layer
W1=rands(16,2);
Wb1=rands(16,1);
%--------------------------1 neuron in ouput layer
W2=rands(1,16);
Wb2=rands(1);
%--------------------------Kth epoch
k=0;
%--------------------------Validation error initializing
E_valid1=100;
%--------------------------Starting Learning Process
while(k < 100 && E_valid1 > epsilon)
    k=k+1;
    for m=1:70
        %--------------------------Feedforward
        %--------------------------1st layer
        net1 = W1 * Xin(:, m) + Wb1;
        O1 = tansig(net1);
        diff_O1 = 4 * exp(-2 .* net1) ./ (1 + exp(-2 .* net1)) .^2;
        %--------------------------output layer
        net2 = W2 * O1 + Wb2;
        O2 = net2;
        diff_O2 = 1;
        %--------------------------Calculating output layer error
        e = D_Learn(:, m) - O2;
        ee(k,m) = e;
        %--------------------------Backpropagation
        %--------------------------output layer
        W2 = W2 + eta * e * O1';
        Wb2 = Wb2 + eta * e;
        %--------------------------1st layer
        e1 = W2' * e;
        delta1 = e1 .* diff_O1;
        W1 = W1 + eta * delta1 * (Xin(:, m))';
        Wb1 = Wb1 + eta *delta1;
    end
    %--------------------------Validation error
    WB1=ones(16,101);
    for p=1:16
    WB1(p,:)=Wb1(p,1)*ones(1,101);
    end
    
    net1_valid = W1 * X_valid + WB1(:,1:20);
    O1_valid = tansig(net1_valid);
 
    WB2 = Wb2 * ones(1,20);
    net2_valid = W2 * O1_valid + WB2(:, 1:20);
    O2_valid = net2_valid;
 
    e_valid = D_valid - O2_valid;
    E_valid = 0.5 * trace(e_valid * e_valid');
    E_v1(k)= E_valid;

    %--------------------------Learning error
    net1_Learn = W1 * Xin + WB1(:,1:70);
    O1_Learn = tansig(net1_Learn);

    WB2 = Wb2 * ones(1,70);
    net2_Learn = W2 * O1_Learn + WB2(:,1:70);
    O2_Learn = net2_Learn;
 
    e_Learn = D_Learn-O2_Learn;
    E_Learn = 0.5 * trace(e_Learn * e_Learn');
    E_L(k) = E_Learn;
end

%--------------------------Plots----------------------------------------%
p = length(E_L);
m = 1:1:p;
figure; 
plot(m, E_L,'g');
hold on;
plot(m, E_v1,'b');
title('Error of Learning (green) and Evaluation 1 (blue) Using Tansig');
xlabel('epoch')
ylabel('Learning and Evaluation 1 Error');

%--------------------------Test----------------------------------------%
WB1 = ones(16,101);
for p=1:16
    WB1(p,:)=Wb1(p,1)*ones(1,101);
end

net1_test=W1*X_test+WB1(:,1:11);
 
O1_test=tansig(net1_test);
 
WB2=Wb2*ones(1,11);
net2_test=W2*O1_test+WB2(:,1:11);
O2_test=net2_test;
 
e_test = D_test-O2_test;
E_test = 0.5*trace(e_test * e_test');









