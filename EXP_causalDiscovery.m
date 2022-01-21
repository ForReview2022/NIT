clc
clear
% [skeleton,names] = readRnet( '.\dataset\cancer.net');
[skeleton,names] = readRnet( '.\dataset\asia.net');
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
nSample = 1000;
SCORE = 0;
TIMESCORE = 0;
ErrorBar = [];
for Times = 1:1000
    Times
    data = SEMDataGenerator(skeleton, nSample, 'uniform', 0.2,0.9,0.1);
    conSize = 1; % conditional set size
    %--------------------------------sklearn_NIT
    tic
    Cskeleton = sklearn_NIT(data,conSize);
    Score1 = ScoreSkeleton(Cskeleton,skeleton)
    Time1 = toc
    %--------------------------------sklearn_NITfg
    tic
    Cskeleton = sklearn_NITfg(data,conSize);
    Score2 = ScoreSkeleton(Cskeleton,skeleton)
    Time2 = toc
    %--------------------------------sklearn_Darling
    tic
    Cskeleton = sklearn_Darling(data,conSize);
    Score3 = ScoreSkeleton(Cskeleton,skeleton)
    Time3 = toc
    %--------------------------------sklearn_ReCIT
    tic
    Cskeleton = sklearn_ReCIT(data,conSize);
    Score4 = ScoreSkeleton(Cskeleton,skeleton)
    Time4 = toc
    %--------------------------------sklearn_FRCIT
    tic
    Cskeleton = sklearn_FRCIT(data,conSize);
    Score5 = ScoreSkeleton(Cskeleton,skeleton)
    Time5 = toc
    %--------------------------------sklearn_NITb
    tic
    Cskeleton = sklearn_NITb(data,conSize);
    Score6 = ScoreSkeleton(Cskeleton,skeleton)
    Time6 = toc
    %--------------------------------sklearn_HSCIT
    tic
    Cskeleton = sklearn_HSCIT(data,conSize);
    Score7 = ScoreSkeleton(Cskeleton,skeleton)
    Time7 = toc
    %--------------------------------sklearn_PaCoT
    tic
    Cskeleton = sklearn_PaCoT(data,conSize);
    Score8 = ScoreSkeleton(Cskeleton,skeleton)
    Time8 = toc
    %--------------------------------result
    temp = [Score1;Score2;Score3;Score4;Score5;Score6;Score7;Score8];
    ErrorBar = [ErrorBar,temp];
    SCORE = SCORE + [Score1;Score2;Score3;Score4;Score5;Score6;Score7;Score8];
    TIMESCORE = TIMESCORE + [Time1;Time2;Time3;Time4;Time5;Time6;Time7;Time8];
    scoreNow = SCORE/Times
    timeNow = TIMESCORE/Times
end