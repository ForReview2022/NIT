clc;
clear;
addpath('.\dataset');addpath('.\hsicAlg');addpath('.\kcitAlg');
tic;
nsamples = 1000;
n = 5; % dimension
m = 2; % num of methods
h = [1 1.2 1.4 1.6 1.8];
%-------------Score Matrix: s1 ~ type I error, s2 ~ type II error
M_TypeI  = zeros(m*length(h),n);
M_TypeII = zeros(m*length(h),n);
for Times = 1:10000 % 1000 times
    TimeIs = Times
    %-------------data generating Case II
    for i = 1:n
        z  = rand(nsamples,i)-0.5;
        ny = rand(nsamples,1)-0.5;
        nx = rand(nsamples,1)-0.5;
        y = 0;
        x = 0;
        for t = 1:i
            x = x + z(:,t);
            y = y + z(:,t);
        end
        for k = 1:length(h)
            Y = h(k)*y/(max(y)-min(y)) + ny/(max(ny)-min(ny)); % 1 1.2 1.4 1.6 1.8
            X = h(k)*x/(max(x)-min(x)) + nx/(max(nx)-min(nx));
            %-------------Type I error
            conset = z;
            Num = i;
            M_TypeI(k,Num) = M_TypeI(k,Num) +  NIT(X,Y,conset);
            M_TypeI(k+length(h),Num) = M_TypeI(k+length(h),Num) +  ReCIT(X,Y,conset);
            %--------------Type II error ----------------------
            randID = randperm(i-1)+1;
            if i == 1
                conset = [];
            else
                conset = z(:,randID(1:i-1));
            end
            M_TypeII(k,Num) = M_TypeII(k,Num) +  NIT(X,Y,conset);
            M_TypeII(k+length(h),Num) = M_TypeII(k+length(h),Num) +  ReCIT(X,Y,conset);
        end
    end
    %-------------Results
    TypeI = 1 - M_TypeI/Times
    TypeII = M_TypeII/Times
    toc
end
toc;