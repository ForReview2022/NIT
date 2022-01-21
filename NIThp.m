function [ind] = NIThp(x,y,z)
n = length(x);
if ~isempty(z)
    M = (eye(size(x,1))-z*((z'*z)^-1)*z');
    res1 = M*x;
    res2 = M*y;
    ind = py.nit.main_nithp(res1,res2);
else
    ind = py.nit.main_nithp(x,y);
end


