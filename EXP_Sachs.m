clc
clear
% nSample = 1000;
addpath('.\dataset');addpath('.\hsicAlg');addpath('.\kcitAlg');
load sachs.mat;load sachs_skeleton.mat;data = sachs;load ind_table.mat; 
% [skeleton,~] = readRnet( '.\dataset\Alarm.net');
%  data = genData(skeleton, nSample);
for i = 1:size(data,2)
    data(:,i) = data(:,i)/(max(data(:,i))-min(data(:,i)));
    data(:,i) = data(:,i) - mean(data(:,i));
end
% plot(digraph(skeleton))
tic
% algM = {@NIT};
algM = {@NIT,@NITfg,@Darling,@ReCIT,@HSCIT,@FRCIT,@PaCoT};
% algM = {@KCIT};
for T = 1:1000
    printT = T
    conSize = 0; % conditional set size 11-2 = 9
    score = [];
    for i = 1:size(algM,2)
%         Cskeleton = PC_sklearn(data,conSize,algM{i});
%         rpf = getRPF(Cskeleton,skeleton);
        rpf = ind_check(data,ind_table,algM{i});
        score = [score;rpf];
    end
    scoreCell{T} = score; % to calculate error bar
    if T == 1
        scoreM = zeros(size(score,1),size(score,2));
    end
    scoreM = [scoreM + score];
    printScore = scoreM/T % average score
    toc
end


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