function [ind] = NIT(x,y,z)
if ~isempty(z)
    M = (eye(size(x,1))-z*((z'*z)^-1)*z');
    res1 = M*x;
    res2 = M*y;
%     ind = PaCoT(res1,res2,[]);
%     if ind
        ind = py.nit.main_nit(res1,res2);
%     end
else
%     ind = PaCoT(x,y,[]);
%     if ind
        ind = py.nit.main_nit(x,y);
%     end
end


