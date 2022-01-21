clear
clc
n = 100;
D = rand(n,10);
e1 = D(:,1) - mean(D(:,1));
e2 = D(:,2) - mean(D(:,2));
e3 = D(:,3) - mean(D(:,3));
e4 = D(:,4) - mean(D(:,4));
x = e1 + e2;
y = e1 - e2;
z = e3 - e4;
%---------------------------------fig1-1
% lw = 1.2;
% [b,a] = ksdensity(x);
% plot(a,b,'k','LineWidth',lw)
% hold on 
% 
% [d,c] = ksdensity(y);
% plot(c,d,'b','LineWidth',lw)
% hold on 
% 
% [d,c] = ksdensity(z);
% plot(c,d,'r','LineWidth',lw)
% grid minor
% 
% legend('\it{x}','\it{y}','\it{z}','FontName','Times New Roman')
% xlabel('\it{\theta}','FontName','Times New Roman');
% ylabel('\it{f(\theta)}','FontName','Times New Roman');

%---------------------------------fig1-2
% lw = 1.2;
% [b,a] = ksdensity(x+y);
% plot(a,b,'b','LineWidth',lw)
% hold on 
% 
% [d,c] = ksdensity(x+z);
% plot(c,d,'r','LineWidth',lw)
% grid minor
% 
% legend('\it{x+y}','\it{x+z}','FontName','Times New Roman')
% xlabel('\it{\theta}','FontName','Times New Roman');
% ylabel('\it{f(\theta)}','FontName','Times New Roman');

%---------------------------------fig1-3
d = (rand(n,10)-0.5)*2; 
x = (d(:,1) + d(:,2)).^2;
y = (d(:,1) - d(:,2));
z = y(randperm(n));
corr(x,y,'type','Pearson')
% -----------------------------------
% lw = 1.2;
% [b,a] = ksdensity(x+y);
% plot(a,b,'b','LineWidth',lw)
% hold on 
% 
% [d,c] = ksdensity(x+z);
% plot(c,d,'r','LineWidth',lw)
% grid minor
% 
% legend('\it{x}^2+\it{y}','\it{x}^2+\it{z}','FontName','Times New Roman')
% xlabel('\it{\theta}','FontName','Times New Roman');
% ylabel('\it{f(\theta)}','FontName','Times New Roman');
%-----------------------------------fig1-4
x = sin(x);
y = tanh((y).^2);
z = y(randperm(n));
c = corr(x,y,'type','Pearson')

lw = 1.2;
[b,a] = ksdensity(x+y);
plot(a,b,'b','LineWidth',lw)
hold on 

[d,c] = ksdensity(x+z);
plot(c,d,'r','LineWidth',lw)
grid minor

legend('\psi(x^2)+ \phi(y)','\psi(x^2)+ \phi(z)','FontName','Times New Roman')
xlabel('\it{\theta}','FontName','Times New Roman');
ylabel('\it{f(\theta)}','FontName','Times New Roman');