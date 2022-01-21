clear
clc
n = 1000; % sample size
d = (rand(n,10)-0.5)*2; % uniform
x = (d(:,1) + d(:,2)).^2;
y = (d(:,1) - d(:,2));
corr(x,y)
z = y(randperm(n));

subplot(1,2,1)
[fa, xa] = ksdensity(x+y);
plot(xa, fa,'b')
hold on
[fb, xb] = ksdensity(x+z);
plot(xb, fb,'r')
hold on


x = sin(x);
y = tanh((y).^2);
z = y(randperm(n));
corr(x,y)
subplot(1,2,2)
[fa, xa] = ksdensity(x+y);
plot(xa, fa,'b')
hold on
[fb, xb] = ksdensity(x+z);
plot(xb, fb,'r')
hold on