clear
clc
tic
n = 1000; % sample size
        B = (rand(n,1)-0.5)*2; % uniform
        subplot(2,2,1)
        [f, x] = ksdensity(B);
        plot(x, f,'.r')
        hold on
%     elseif dis_type == 2
        B = chi2rnd(9,n,1); % chi2rnd
        subplot(2,2,2)
        [f, x] = ksdensity(B);
        plot(x, f,'.b')
        hold on
%     elseif dis_type == 3
        B = exprnd(1.5,n,1); % exprnd
        subplot(2,2,3)
        [f, x] = ksdensity(B);
        plot(x, f,'.k')
        hold on
%     elseif dis_type == 4
        B = betarnd(1,0.2,n,1); % beta
        subplot(2,2,4)
        [f, x] = ksdensity(B);
        plot(x, f,'.g')
        hold on
%     end
% end
