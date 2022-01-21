clear;
clc;
tic
% load normalized_Leukemia.mat
% load normalized_Colorectal.mat
% load normalized_Liver.mat
load normalized_Leukemia_ATL.mat 
data = d;
n = size(data,2)-1;
x = data(:,n+1);
data = data(:,1:n);
non = []; %non-adjacent
%---------------------------- 0-order CI tests-----------------------------
fprintf('--------------- 0-order CI tests \n');
for i = 1:n
    ind1 = PaCoT(x,data(:,i),[]);
    if ind1 
        ind2 = NIT(x,data(:,i),[]);
        if ind2
            non = [non,i];
        end
    end
end
non1 = non;
%---------------------------- 1-order CI tests-----------------------------
fprintf('--------------- 1-order CI tests \n');
idx1 = setdiff(1:n,non);
len1 = length(idx1);
for j = 1: len1
    [j,len1]
    for k = 1: len1
        if j~=k && isempty(intersect(idx1(k),non))
            y = data(:,idx1(j));
            z = data(:,idx1(k));
            ind1 = PaCoT(x,y,z);
            if ind1
                try
                    xf = fit_gpr(z,x,cov,hyp,Ncg);
                    res1 = xf-x;
                    yf = fit_gpr(z,y,cov,hyp,Ncg);
                    res2 = yf-y;
                    ind2 = NIT(res1, res2,[]);
                    if ind2
                        non = [non,idx1(j)];
                        break;
                    end
                catch
                    non = [non,idx1(j)];
                    break;
                end
            end
        end
    end
end
found_Genes = unique(setdiff(1:n,non))
toc