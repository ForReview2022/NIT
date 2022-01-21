clc;
clear;
tic
nsamples = 1000;
n = 5; % dimension of Z
m = 6; % num of methods
%-------------Score Matrix: s1 ~ type I error, s2 ~ type II error
M_TypeI = zeros(m,n);
M_TypeII = zeros(m,n);
%-------------main test
for Times = 1:10000000 % iter times
    TimeIs = Times
    %-------------data generating Case I
    z = (rand(nsamples,n)*2-1);
    z = z - mean(z);
    a = rand*0.8+0.2;
    b = rand*0.8+0.2;
    n1 = (rand(nsamples,1)*2-1)*1;
    n2 = (rand(nsamples,1)*2-1)*1;
    n1 = n1 - mean(n1);
    n2 = n2 - mean(n2);
    x = z(:,1);
    z(:,1) = a*x + n1;
    y = b*z(:,1) + n2;
    %-------------Type II error
    randID = randperm(4)+1;
    for Num = 1:n
        if Num == 1
            conset = [];
        else
            conset = z(:,randID(1:Num-1));
        end
        ind  = PaCoT(x,y,conset);
        if  ind
            M_TypeII(1,Num) = M_TypeII(1,Num) +  NIT(x,y,conset)+1000000;
            M_TypeII(2,Num) = M_TypeII(2,Num) +  Darling(x,y,conset)+1000000;
            M_TypeII(3,Num) = M_TypeII(3,Num) +  NITfg(x,y,conset)+1000000;
            M_TypeII(4,Num) = M_TypeII(4,Num) +  NITb(x,y,conset)+1000000;
            M_TypeII(5,Num) = M_TypeII(5,Num) +  FRCIT(x,y,conset)+1000000;
            M_TypeII(6,Num) = M_TypeII(6,Num) +  ReCIT(x,y,conset)+1000000;
        end
    end
    %-------------Results
    TypeII = M_TypeII
    toc
end