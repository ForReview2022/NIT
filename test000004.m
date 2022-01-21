clear;clc;addpath('.\dataset');addpath('.\hsicAlg');addpath('.\kcitAlg');
% n = 2000;
% x = rand(n,1);
% z = x + rand(n,1)*0.2;
% y = z + rand(n,1)*0.2;
% x = x - mean(x);
% y = y - mean(y);
% z = z - mean(z);
% M = (eye(size(x,1))-z*((z'*z)^-1)*z');
% res1 = M*x;
% res2 = M*y;
% ind = HSCI(res1,res2)
% ind1 = HSCI(res1,z)
% ind2 = HSCI(res2,z)
load sim1.mat
% imagesc(squeeze(mean(net)))
d = ts(1:2000,:)
% Cskeleton = PC_sklearn(d,2,@ReCIT)
% plot(graph(Cskeleton,'upper'))

% z ---> x
x = d(:,2);
z = d(:,3);
x = x - mean(x);
z = z - mean(z);
M = (eye(size(x,1))-z*((z'*z)^-1)*z');
res = M*x;
ind1 = HSCIT(res,z,[])
%-------------------------------------
% z ---> x
x = d(:,3);
z = d(:,2);
x = x - mean(x);
z = z - mean(z);
M = (eye(size(x,1))-z*((z'*z)^-1)*z');
res = M*x;
ind2 = HSCIT(res,z,[])