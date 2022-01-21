clc;
clear;
tic
nsamples = 1000;
n = 10; % dimension of Z
m = 3; % num of methods
%-------------Score Matrix: type II error
M_TypeII = zeros(m,n/2);
%-------------main test
for Times = 1:10000 % iter times
    TimeIs = Times
    %-------------data generating Case I
    z = rand(nsamples,n)-0.5;
    a = rand*0.8+0.2;
    b = rand*0.8+0.2;
    n1 = rand(nsamples,1)-0.5;
    n2 = rand(nsamples,1)-0.5;
    c = z(:,1);
    x = 0.35*c + n1;  % 0.3  0.35
    y = 0.35*c + n2;
    %-------------Type II error
    randID = randperm(n-1)+1;
    for Num = 2:2:n
        conset = z(:,randID(1:Num-1));
        idx = Num/2;
        M_TypeII(1,idx) = M_TypeII(1,idx) +  NIT(x,y,conset);
        M_TypeII(2,idx) = M_TypeII(2,idx) +  Darling(x,y,conset);
        M_TypeII(3,idx) = M_TypeII(3,idx) +  ReCIT(x,y,conset);
    end
    %-------------Results
    TypeII = M_TypeII/Times
    toc
end