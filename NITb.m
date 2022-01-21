function [ind] = NITb(x,y,z)
if ~isempty(z)
    M = (eye(size(x,1))-z*((z'*z)^-1)*z');
    res1 = M*x;
    res2 = M*y;
    ind = py.nitb.main_nitb(res1,res2);
else
    ind = py.nitb.main_nitb(x,y);
end


