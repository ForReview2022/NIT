function [ind] = NITfg(x,y,z)
if ~isempty(z)
    M = (eye(size(x,1))-z*((z'*z)^-1)*z');
    res1 = M*x;
    res2 = M*y;
    ind = py.nitfg.main_nitfg(res1,res2);
else
    ind = py.nitfg.main_nitfg(x,y);
end


