%%%%%%%%%%%%%% Various nonlinear functions used in the model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f = F(x,type,sigma,f_max,N,shift) 
%F(P): Steady-state firing rate of the neuron when the input is P, and P is time-independent.
if type ==1
    %% Sigmoidal
    f = f_max./(1+exp(-N*(x-shift)));
elseif type ==2
    %% Naka-Rushton function
    if x>=0
        f = (f_max*x.^N)./(sigma^N+x.^N);
    else
        f = 0;
    end
    
elseif type ==3
    %% Linear
    f = x;
elseif type ==4
    %% Half-wave rectification
    if x>=shift
        f =1*x;%2*x;
    else
        f = 0;
    end
end

end