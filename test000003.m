clear;clc;
n = 1000;
% B = (rand(n,1)-0.5)*2; % uniform
% B = chi2rnd(1,n,1); % chi2rnd
% B = exprnd(0.5,n,1); % exprnd
B = betarnd(0.2,0.8,n,1); % beta
lw = 1.2;
[b,a] = ksdensity(B);
plot(a,b,'b','LineWidth',lw)
hold on 
grid minor
% legend('\psi(x^2)+ \phi(y)','\psi(x^2)+ \phi(z)','FontName','Times New Roman')
% xlabel('\it{\theta}','FontName','Times New Roman');
% ylabel('\it{f(\theta)}','FontName','Times New Roman');