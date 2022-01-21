function [ind] = NITfgn(x,y,z)
if ~isempty(z)
    xf = fit_gpr(z,x,cov,hyp,Ncg);
    res1 = xf-x;
    yf = fit_gpr(z,y,cov,hyp,Ncg);
    res2 = yf-y;
    ind = py.nitfg.main_nitfg(res1,res2);
else
    ind = py.nitfg.main_nitfg(x,y);
end


