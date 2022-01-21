clear;clc;
% [skeleton,names] = readRnet( '.\dataset\cancer.net');
% [skeleton,names] = readRnet( '.\dataset\asia.net');
% [skeleton,names] = readRnet( '.\dataset\child.net');
% [skeleton,names] = net2sketelon( '.\dataset\insurance.net');
% [skeleton,names] = net2sketelon( '.\dataset\Alarm.net');
%  [skeleton,names] = readRnet( '.\dataset\barley.net');
% [skeleton,names] = net2sketelon( '.\dataset\hailfinder.net');
% [skeleton,names] = net2sketelon( '.\dataset\win95pts.net');
% [skeleton,names] = readRnet( '.\dataset\pathfinder.net');
% [skeleton,names] = readRnet( '.\dataset\andes.net');
% [skeleton,names] = hugin2skeleton( '.\dataset\Pigs.hugin');
skeleton = sortskeleton(skeleton);
% G1 = digraph(skeleton);
% h = plot(G1);
skeleton = sortskeleton(skeleton);
nSample = 1000; % sample size 20 ~ 40
data = genData(skeleton, nSample);
% skeleton
Cskeleton = PC_sklearn(data,3,@PaCoT);
getRPF(Cskeleton,skeleton)
ratio = sum(sum(Cskeleton))/(size(skeleton,2)^2)
true_ratio = sum(sum(skeleton))/(size(skeleton,2)^2)

function[data]=genData(skeleton,nsamples)
[dim, ~]=size(skeleton);
data = rand(nsamples, dim)*2-1;
for k = 1:dim
    data(:,k) = data(:,k) - mean(data(:,k));
end
for i=1:dim
    parentidx=find(skeleton(:,i)==true);
    for j=1:length(parentidx)
        if parentidx(j)==i
            parentidx(j)=[];
        end
    end
    if ~isempty(parentidx)
        pasample = 0;
        for w = 1:length(parentidx)
            pasample = pasample + data(:, parentidx(w))*(rand*0.8+0.2);
        end
        n =  (rand(nsamples,1)*2-1)*0.2;
        n = n - mean(n);
        data(:, i)= pasample + n;
    end
end
end