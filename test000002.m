clear;clc;
load temp.mat
temp = realtime_error_rate_ind
idx = [1,6,11,16];
a = sum(temp(:,idx),2)/5;
b = sum(temp(:,idx+1),2)/5;
c = sum(temp(:,idx+2),2)/5;
d = sum(temp(:,idx+3),2)/5;
e = sum(temp(:,idx+4),2)/5;
f = [a,b,c,d,e];
g = [f,mean(f,2)]