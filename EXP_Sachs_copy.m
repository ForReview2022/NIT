clc
clear
addpath('.\dataset');addpath('.\hsicAlg');addpath('.\kcitAlg');
load sachs.mat;load sachs_skeleton.mat;data = sachs;
% for i = 1:size(data,2)
%     data(:,i) = data(:,i)/(max(data(:,i))-min(data(:,i)));
%     data(:,i) = data(:,i) - mean(data(:,i));
% end
% data = (mapstd(sachs'))'; % zero mean and unit variance
tic
algM = {@NIT,@NITfg,@NITb,@SCITn,@SCIT,@Darling,@FRCIT,@PaCoT,@ReCIT,@HSCIT};
for T = 1:100
    printT = T
    conSize = 0; % conditional set size 11-2 = 9
    score = [];
    for i = 1:size(algM,2)
        Cskeleton = PC_sklearn(data,conSize,algM{i});
        rpf = getRPF(Cskeleton,skeleton);
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