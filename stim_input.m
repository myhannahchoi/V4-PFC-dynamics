%%%%%%%%%%%%% Generate stimulus inpust to V4 units %%%%%%%%%%%%%%%%%%%%%%%%
function y = stim_input(x,t_start,t_end,amp)
global dt
global display_fig

y0 = step(x,t_start,t_end,amp);
x_filter = [-x(end)/2:dt:x(end)/2];
filter1 = gaussian(x_filter,0,10,1,0);
filter2 = 1*DoG(x_filter,30,50,20,20,15,10,0);
filter1 = kernel_norm(filter1);
filter2 = kernel_norm(filter2);
z1 = conv(filter1,y0,'same');
z3 = z1;
y2 = conv(filter2,y0,'same');
y3 = y2.^3/(max(y2)^2);
ii_neg = find(y3<0);
y3(ii_neg) = 0;

if display_fig == 1
    figure(14), hold on, subplot(2,1,1), plot(x,y0,'k')
    figure(14), hold on, subplot(2,1,2), plot(x,y3,'k')
end

y = y3;

end

%% Input Stimuli
function y1 = step(x1,t_start,t_end,amp)
slope = -0.05;
y1 = zeros(1,length(x1));
for ii = 1:length(x1)
    if (x1(ii) >=t_start) && (x1(ii) <t_end)
        y1(ii) = amp+x1(ii)*slope;
    end
end

i_neg = find(y1<0);
y1(i_neg) =0;
end

%% Filters
function y = gaussian(x,mu,sig,a,d)
y = a*exp((-(x - mu).^2)./(2 * sig^2))+d;
end

function dz_x = dx_gaussian(x,mu,sig,peak,d)
temp = abs((exp(-((x-mu).^2)/(2*sig^2)).*(-(x-mu)./(sqrt(2*pi)*sig^3))));
a = 1/max(max(temp));
factor = peak*a;

A = 2;
dz_x = factor*(exp(-((x-mu).^2)/(2*sig^2)).*(-(A*x-mu)./(sqrt(2*pi)*sig^3)))+d;

end

function y = biphasic(x,mu,sig,a,b)
y = (a*(x-mu)+b).*exp(-((x-mu).^2)/(2*sig^2));
end

function y = DoG(x,mu1,mu2,sig1,sig2,a1,a2,d)
y1 = a1*exp((-(x - mu1).^2)./(2 * sig1^2))+d;
y2 = a2*exp((-(x - mu2).^2)./(2 * sig2^2))+d;
y = y1-y2;

end

function gb = GB(x,sig,lambda,psi)
gb= exp(-.5*(x.^2/sig^2)).*cos(2*pi/lambda*x+psi);
end

%% Normalization
function y = kernel_norm(v)
%%% The Standard Kernel Normalization
y = v./sum(sum(abs(v)));
end

function y = peak_norm(v)
y = v./max(abs(v));
end