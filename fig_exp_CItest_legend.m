clear
clc
lw = 1.2;
load Case2TypeII.mat
d = TypeII
x = 1:5
nit1 = d(1,:);
nit2 = d(2,:);
nit3 = d(3,:);
nit4 = d(4,:);
nit5 = d(5,:);
k1 = d(6,:);
k2 = d(7,:);
k3 = d(8,:);
k4 = d(9,:);
k5 = d(10,:);
plot(x,nit1,'-dr','LineWidth',lw)
hold on
plot(x,k1,'-db','LineWidth',lw)
hold on
plot(x,nit2,'-.+r','LineWidth',lw)
hold on
plot(x,k2,'-.+b','LineWidth',lw)
hold on
plot(x,nit3,'--or','LineWidth',lw)
hold on
plot(x,k3,'--ob','LineWidth',lw)
hold on
plot(x,nit4,'-.*r','LineWidth',lw)
hold on
plot(x,k4,'-.*b','LineWidth',lw)
hold on
plot(x,nit5,':sr','LineWidth',lw)
hold on
plot(x,k5,':sb','LineWidth',lw)
hold on
% grid minor
xticks([1 2 3 4 5])
set(gca,'Fontname', 'Times New Roman')
yticks([0 0.2 0.4 0.6 0.8])
legend('NIT-a_1','KCIT-a_1','NIT-a_2','KCIT-a_2','NIT-a_3','KCIT-a_3','NIT-a_4','KCIT-a_4','NIT-a_5','KCIT-a_5','NumColumns',5)
xlabel('Dimension of \itZ','FontName','Times New Roman','FontSize',20);
ylabel('Type II error rate','FontName','Times New Roman','FontSize',20);