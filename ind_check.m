function [rpf] = ind_check(data,d,Alg)
n = size(d,1);
ind_num = sum(d(:,3))
nonind_num = n - ind_num
temp1 = 0;
temp2 = 0;
% tempM = [];
for i = 1:n
    x = data(:,d(i,1));
    y = data(:,d(i,2));
    ind = Alg(x,y,[]);
%     tempM = [tempM;[d(i,1),d(i,2),ind]];
    if ind ~= d(i,3)
        if d(i,3) == 1
            temp1 = temp1 + 1;
        else
            temp2 = temp2 + 1;
        end
    end
end
a = temp1/ind_num;
b = temp2/nonind_num;
rpf = [a,b,mean([a,b])];
% tempM
end