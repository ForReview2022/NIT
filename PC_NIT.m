function [Cskeleton] = PC_NIT(data,consetNum)
n = size(data,2);
alpha = 0.05;
Cskeleton=zeros(n,n);
Acell = cell(1,n);
Bcell = cell(1,n);
Ccell = cell(1,n);
MN = nchoosek(1:n,2);
for p = 1:size(MN,1)
    Acell{p}=-1;
    Bcell{p}=-1;
    Ccell{p}=-1;
end
for i= 1:(n-1)
    lab = i;
    pc = [];
    for j = i+1:n
        flag = true;
        indf = NIT(data(:,i), data(:,j),[]);
        if indf == false
            temp = (1:n);
            temp([i,j])=[];
            conSepSet = temp;
            for k=1:min(consetNum,length(conSepSet))
                if flag == false
                    break;
                end
                condPaAll = nchoosek(temp,k);
                numCondPaAll = size(condPaAll,1);
                for p = 1:numCondPaAll
                    condPa = condPaAll(p,:);
                    x = data(:,i);
                    y = data(:,j);
                    z = data(:, condPa);
                    ind = NIT(x,y,z);
                    if ind 
                        flag = false;
                        break;
                    end
                end
            end
            if flag == true
                pc = unique([pc,j]);
            end
        end
    end
    len=length(pc);
    for w = 1:len
        Cskeleton(lab,pc(w))=1;
        Cskeleton(pc(w),lab)=1;
    end
end