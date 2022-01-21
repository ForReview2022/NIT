function [ind] = HybridT(x,y,z)
if ~isempty(z)
    M = (eye(size(x,1))-z*((z'*z)^-1)*z');
    res1 = M*x;
    res2 = M*y;
    ind = HT(res1,res2);
else
    ind = HT(x,y);
end

function [ind] = HT(x,y)
ind = 0;
if PaCoT(x,y,[])
    if NIT(x,y,[])
        if NITfg(x,y,[])
            if NITb(x,y,[])
                if Darling(x,y,[])
                    if FRCIT(x,y,[])
                        if KCIT(x,y,[])
                            if HSCIT(x,y,[])
                                ind = 1;
                            end
                        end
                    end
                end
            end
        end
    end
end



