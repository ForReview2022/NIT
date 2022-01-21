%Gretton and Gyorfi (2008) L1 Test

%Equiprobable space partitioning (c) Kenji Fukumizu  

%Arthur Gretton
%24/01/08



%Inputs: 
%        X contains dx columns, L rows. Each row is an i.i.d sample
%        Y contains dy columns, L rows. Each row is an i.i.d sample
%        params.q is number of partitions per dimension (Ku and Fine use 4),


%Outputs: 
%        thresh: test threshold for level alpha test
%        testStat: test statistic

function [thresh,testStat] = GreGyoL1Test(X,Y,alpha,params);

q=params.q;

L = size(X,1);
dx = size(X,2);
dy = size(Y,2);


[idxX,blockX] = EqualPartition(X,q);
[idxY,blockY] = EqualPartition(Y,q);

blockX = [cumsum(blockX)-blockX(1) sum(blockX)];
blockY = [cumsum(blockY)-blockY(1) sum(blockY)];

testStat = 0;


for indCellX = 1:q^dx
  for indCellY = 1:q^dy

    indSet = intersect(idxX(blockX(indCellX)+1:blockX(indCellX+1)),idxY(blockY(indCellY)+1:blockY(indCellY+1)));
    Nj = length(indSet);
    testStat = testStat + abs(Nj/L - 1/q^dx/q^dy);


  end
  
 
end


%DEGREES OF FREEDOM: the next line assumes dx=dy
%Thus, both X and Y are partitioned into q^dx bins
thresh = sqrt(2*q^dx*q^dy/pi/L) + (1-2/pi)/sqrt(L)*icdf('normal',1-alpha,0,1);




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% EqualPartition()
% Recursive partitioninig of the dimension with the equal number of data.
%
% Date: June 1, 2007
% Version: 0.9
% Author: Kenji Fukumizu
% Affiliation: Institute of Statistical Mathematics, ROIS
% (c) Kenji Fukumizu
%
% Input:
% X - data matrix (row=data, column=dimension)
% K - Number of partitions for each dimension
% (Total number of bins = K^{dim X})
% Output:
% idx - sorted index of X in the order of the bins
% block - the number of data in the bins
%
% X(idx(block(j-1)+1:block(j)),:) are the data in j-th bin (for j>=2)
%

function [idx block] = EqualPartition(X,K)
%function [cellCount,partitionInd]=getCellCount(sample,q)

[N dim]=size(X);
idx=1:N;
prevblock=[N];
for h=1:dim
    numblk=length(prevblock);
    start=1;
    b_end=cumsum(prevblock,2);
    tmpblock=[];
    
    for i=1:numblk
        subidx=idx(start:b_end(i));
        [v is]=sort(X(subidx,h));
        nk=length(subidx);
        num=floor(nk/K);
        if num<=1
            error('Use smaller number of partitions (K): EqualPartition()');
            % Each bin must contain at least two data.
        end
        subblk=num*ones(1,K)+[zeros(1,K-mod(nk,K)) ones(1,mod(nk,K))];
        tmpblock=[tmpblock subblk];
        idx(start:b_end(i))=subidx(is);
        start=b_end(i)+1;
    end
    prevblock=tmpblock;
end
block=tmpblock;





