%This function implements the HSIC test using a bootstrap approximation
%to the test threshold

%Arthur Gretton
%31/05/07


%Inputs:
%        X contains dx columns, m rows. Each row is an i.i.d sample
%        Y contains dy columns, m rows. Each row is an i.i.d sample
%        alpha is the level of the test
%        params.bootForce=1: do bootstrap. If 0: only bootstrap if no
%        previous threshold file found
%        params.shuff is number of shuffles to approximate null distribution
%        params_sigx is kernel size for x (set to median distance if -1)
%        params_sigy is kernel size for y (set to median distance if -1)

%Outputs:
%        thresh: test threshold for level alpha test
%        testStat: test statistic

%Set kernel size to median distance between points, if no kernel specified

%04/06/07 Faster computation of the dependence measure (without using low rank)
%         Compute median distance using 100 points to save time

%12/02/08 Allow use of previously saved test thresholds

function [ind] = HSCI(X,Y)
alpha = 0.05;
params_sigx = -1;
params_sigy = -1;

m=size(X,1);
dx=size(X,2);

%Set kernel size to median distance between points, if no kernel specified
if params_sigx == -1
    size1=size(X,1);
    if size1>100
        Xmed = X(1:100,:);
        size1 = 100;
    else
        Xmed = X;
    end
    G = sum((Xmed.*Xmed),2);
    Q = repmat(G,1,size1);
    R = repmat(G',size1,1);
    dists = Q + R - 2*Xmed*Xmed';
    dists = dists-tril(dists);
    dists=reshape(dists,size1^2,1);
    params_sigx = sqrt(0.5*median(dists(dists>0)));  %rbf_dot has factor of two in kernel
end

if params_sigy == -1
    size1=size(Y,1);
    if size1>100
        Ymed = Y(1:100,:);
        size1 = 100;
    else
        Ymed = Y;
    end
    G = sum((Ymed.*Ymed),2);
    Q = repmat(G,1,size1);
    R = repmat(G',size1,1);
    dists = Q + R - 2*Ymed*Ymed';
    dists = dists-tril(dists);
    dists=reshape(dists,size1^2,1);
    params_sigy = sqrt(0.5*median(dists(dists>0)));
end

%Compute the Gram matrices
K = rbf_dot(X,X,params_sigx);
L = rbf_dot(Y,Y,params_sigy);

%Biased terms
bone = ones(m,1);
H = eye(m)-1/m*ones(m,m);

Kc = H*K*H;
testStat = 1/m^2*sum(sum(Kc'.*L));
threshFileName = strcat('hsicTestThresh',num2str(m),'_',num2str(dx));
params.bootForce = 1;
if ~exist(strcat(threshFileName,'.mat'),'file')  || params.bootForce==1

    params.shuff = 100;
    HSICarr = zeros(params.shuff,1);
    for whichSh=1:params.shuff

        [~,indL] = sort(rand(m,1));
        HSICarr(whichSh) = 1/m^2*sum(sum(Kc'.*L(indL,indL)));

    end

    HSICarr = sort(HSICarr);
    thresh = HSICarr(round((1-alpha)*params.shuff));
end
if testStat <= thresh
    ind = 1;
else
    ind = 0;
end
end