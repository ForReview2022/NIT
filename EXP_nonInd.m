clear
clc
addpath('.\dataset');addpath('.\hsicAlg');addpath('.\kcitAlg');
tic
n = 2000; % sample size
num_dis_type = 4;
num_func_type = 5;
error_rate_nonind = zeros(num_dis_type*num_func_type,7);
for t = 1:10000 % iter num
    iter = t
    for dis_type = 1:num_dis_type
        if dis_type == 1
            B = (rand(n,10)-0.5)*2; % uniform
        elseif dis_type == 2
            B = chi2rnd(1,n,10); % chi2rnd
        elseif dis_type == 3
            B = exprnd(0.5,n,10); % exprnd
        elseif dis_type == 4
            B = betarnd(0.2,0.8,n,10); % beta
        end
        B = B./(max(B)-min(B)); % normalize
        A = B - mean(B); % 0 mean
        for func_type = 1:num_func_type
            if func_type == 1
                d1 = A(:,1) + A(:,2) + A(:,7) + A(:,8);
                d2 = A(:,1) - A(:,2) + A(:,9);
            elseif func_type == 2
                d1 = sin(A(:,1) + A(:,2)) + A(:,7) + A(:,8);
                d2 = sin(A(:,1) - A(:,2)) + A(:,9);
            elseif func_type == 3
                d1 = exp(A(:,1) + A(:,2)) + A(:,7) + A(:,8);
                d2 = exp(A(:,1) - A(:,2)) + A(:,9);
            elseif func_type == 4
                d1 = log((A(:,1) + A(:,2)).^2) + A(:,7) + A(:,8);
                d2 = log((A(:,1) - A(:,2)).^2) + A(:,9);
            elseif func_type == 5
                d1 = (A(:,1) + A(:,2)).^2 + A(:,7) + A(:,8);
                d2 = (A(:,1) - A(:,2)).^2 + A(:,9);
            end
            d1 = d1 - mean(d1);
            d2 = d2 - mean(d2);
            Num1 =  NIT(d1,d2,[]);
            Num2 =  NITfg(d1,d2,[]);
            Num3 =  Darling(d1,d2,[]);
            Num4 =  KCIT(d1,d2,[]);
            Num5 =  HSCIT(d1,d2,[]);
            Num6 =  FRCIT(d1,d2,[]);
            Num7 =  PaCoT(d1,d2,[]);
            arrayid = (dis_type-1)*num_func_type + func_type;
            error_rate_nonind(arrayid,:) = error_rate_nonind(arrayid,:) + [Num1,Num2,Num3,Num4,Num5,Num6,Num7];
        end
    end
    realtime_error_rate_nonind = [error_rate_nonind/t;sum(error_rate_nonind/t)/(num_dis_type*num_func_type)];
%     save('realtime_error_rate_nonind.mat','realtime_error_rate_nonind')
    [a,b,c,d] = latexPrint(realtime_error_rate_nonind);
    a
    b
    c
    d
    toc

end



function[a,b,c,d] = latexPrint(realtime_error_rate_nonind)
data = realtime_error_rate_nonind;
a = data(1:5,:);
b = data(6:10,:);
c = data(11:15,:);
d = data(16:20,:);
a = [a;mean(a)];
b = [b;mean(b)];
c = [c;mean(c)];
d = [d;mean(d)];
end