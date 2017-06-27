%%%%%%%%----------- V4-vlPFC Simple Network Model -----------%%%%%%%%%%
%%%%%%%%----------- Hannah Choi (hannahch@uw.edu) -----------%%%%%%%%%%
%%%%%%%%-----------          6/27/2017            -----------%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%       This is the main run file generating figures.           %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear all
close all
clc

global dt
global display_fig

%% Setting the parameter values

%%%%%%%%%%%%%%%%%%%%%%%%% Synaptic Weights %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% To use specific values for synaptic weights, uncomment & change 
% ff_same = 0.3;
% fb_same = 0.9;
% PFC_mutual = 0;
% V4_self = 0;
% V4_mutual = 0;
% ff_cross = 0.2;
% ff_same = 0.5;
% fb_same = 0.9;
% ff_cross = 0.2;
PFC_mutual = 0;
V4_self = 0;
V4_mutual = 0;

%%%%%%%%%%%%%%%%%%%%%%%%% Synaptic Delays %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% To use specific values for synaptic weights, uncomment & change 
% syn_latency_uv = 40;
syn_latency_vu = 40;


%%%%%%%%%%%%%%%%%%%%%%%%% Parameter Sweeps %%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig_type = 1; % 1 for the parameters used for the example model cell; 

if fig_type == 1
    %%% Example Cell %%%
    sweep_syn_delay = 40;
    sweep_ff_same = 0.5;
    sweep_ff_cross = 0.2;
    sweep_fb_same = 1.3;
    sweep_fb_cross = sweep_fb_same.*sweep_ff_cross./sweep_ff_same;    
elseif fig_type ==2
    %%% Population Average %%%
    sweep_syn_delay = 20:10:80;
    sweep_ff_same = 0.4:0.1:0.6;
    sweep_ff_cross = 0:0.1:0.2;
    sweep_fb_same = 0.1:0.2:1.3;
    sweep_fb_cross = 0;    
elseif fig_type ==3
    %%% PFC selectivity %%%
    sweep_syn_delay = 40;
    sweep_ff_same = 0;% Do below: 0.7-sweep_ff_cross;
    sweep_ff_cross = [0:0.1:0.3];
    sweep_fb_same = 1;
    sweep_fb_cross = 0;   
elseif fig_type ==4
    %%% V4 peak numbers %%%
    sweep_syn_delay = 40; % or 20 & 80
    sweep_ff_same = 0.5;
    sweep_ff_cross = 0.2;    
    sweep_fb_same = 1.3; % or 1.5
    sweep_fb_cross = sweep_fb_same.*sweep_ff_cross./sweep_ff_same;    
end

total_sweep = length(sweep_syn_delay)*length(sweep_fb_same)...
    *length(sweep_ff_cross)*length(sweep_ff_same)*length(sweep_fb_cross);

%%%%%%%%%%%%%%%%%%%%%%%%% Model Mechanisms %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
syn_adapt = 1; %1: include synaptic adaptation, 0: remove synaptic adaptation
half_wave = 1; %1: include half-wave, 0: remove half-wave rectification
gain_delay  = 0; % 1 to delay gain or 0 to have instant gain
gain_delay_amount = 0; % Delay on the gain modulation, assuming it's coming from IT [10, 15, 20, 50, 80] (ms)
whitenoise = 1; % 1: include noise, 0: no noise
sig = 30; % White noise sigma
selectivity_type = 1; 

%%%%%%%%%%%%%%%%%%%%%%%%% Figure display & saving %%%%%%%%%%%%%%%%%%%%%%%%
display_fig = 0; % 1: display figures during running; 0: no display
fig_peakstar = 0; % 1: place stars on the extrema, 0: no stars

%%%%% Normalization and Gain modulations
normalize = 0; %1: the response is divided by the population activity, 0: not divided
poisson = 0; %1: Generate Poisson spikes, 0: no Poisson spikes and based on firing rates
gain = 4; % Gain modulation type
NL = 2; % Nonlinear filter type

%%%%%%%%%%%%%%%%%%%%%%%%% Naka-Rushton function parameters %%%%%%%%%%%%%%%%%%%
sigma1 = 90;  % Half-maximum point
f_max1 = 100; % Maximum firing rate
N1 = 2; % Slope
sigma2 = 90; % Half-maximum point
f_max2 = 100; % Maximum firing rate
N2 = 2; % Slope

%%%%%%%%%%%%%%%%%%%%%%%% Numerical diffEq solver parameters %%%%%%%%%%%%%%%%%%%%
dt = 0.01;
T = 700;
T_grid = [0:dt:T];
IC = [0 0 0 0]; % Initial firing rate: [v1_0 v2_0 u1_0 u2_0]

%%%%%%%%%%%%%%%%%%%%%%%% Visual input stimuli %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
shape_on = 30; %(ms)
shape_off = 530; %(ms)
percent_unoccl = [100 95 82 72 59];
C = [[0,0,0];[0 0.7 1];[0.7 0.8 0];[1 0.8 0];[1,0,0]];


%% Generate model neuronal responses
i_sweep = 0;

v1_sweep = zeros(total_sweep,length(T_grid),length(percent_unoccl));
v2_sweep = zeros(total_sweep,length(T_grid),length(percent_unoccl));
u1_sweep = zeros(total_sweep,length(T_grid),length(percent_unoccl));
u2_sweep = zeros(total_sweep,length(T_grid),length(percent_unoccl));

selectivity_V4_1st_sweep = zeros(1,total_sweep);
selectivity_V4_2nd_sweep  = zeros(1,total_sweep);
selectivity_PFC_sweep  = zeros(1,total_sweep);


for i_sweep1 = 1:length(sweep_syn_delay)
    syn_latency_uv = sweep_syn_delay(i_sweep1);
    for i_sweep2 = 1:length(sweep_fb_same)
        fb_same = sweep_fb_same(i_sweep2);
        for i_sweep3 = 1:length(sweep_ff_cross)
            ff_cross = sweep_ff_cross(i_sweep3);
            for i_sweep4 = 1:length(sweep_ff_same)
                ff_same = sweep_ff_same(i_sweep4);
                for i_sweep5 = 1:length(sweep_fb_cross)
                    fb_cross = sweep_fb_cross(i_sweep5);
 
                    if  fig_type ==3
                        ff_same =  0.7-ff_cross; %% For PFC selectivity modulation
                    end
                    
                    if (fig_type == 2) || (fig_type ==3)
                        fb_cross = fb_same *ff_cross/ff_same;
                    end

                    close(figure(3))

                    %%%%%%%%%%%%%%%%%     Synaptic weights     %%%%%%%%%%%%%%%%%%%%%%%%     
                    w0_v1u1 = ff_same;
                    w0_v2u2 = ff_same;
                    w0_u1v1 = fb_same;
                    w0_u2v2 = fb_same;
                    w0_u1u2 = PFC_mutual;
                    w0_u2u1 = PFC_mutual;
                    w0_v1v1 = V4_self;
                    w0_v2v2 = V4_self;
                    w0_v1v2 = V4_mutual;
                    w0_v2v1 = V4_mutual;
                    w0_v1u2 = ff_cross;
                    w0_v2u1 = ff_cross;
                    w0_u1v2 = fb_cross;
                    w0_u2v1 = fb_cross;
                    
                    %%%%%%%%%%%% Generate multiple trials to construct histograms %%%%%%%%%%
                    trial = 1;
                    v1_aveFR1 = zeros(trial,length(percent_unoccl));
                    v1_aveFR2 = zeros(trial,length(percent_unoccl));
                    v2_aveFR1 = zeros(trial,length(percent_unoccl));
                    v2_aveFR2 = zeros(trial,length(percent_unoccl));
                    u1_aveFR = zeros(trial,length(percent_unoccl));
                    u2_aveFR = zeros(trial,length(percent_unoccl));
                    
                    for trialnum = 1:trial
                        
                        %%%%%%%%%%%% Input stimuli to V4 %%%%%%%%%%%%
                        stim_occl = 1*(20+2.5*percent_unoccl.^(1)); %Stimulus input to preferred shape V4
                        stim_occl_np = 80+12*percent_unoccl.^(1/3); %Stimulus input to non-preferred shape
                        
                        %%%%%% Threshold firing rates %%%%%%
                        r_th_v1 = 20; 
                        r_th_v2 = 20; 
                        r_th_u1 = 0; 
                        r_th_u2 = 0; 
                        
                        %%%%%% Thime constants %%%%%%%%%%%%%
                        tau1 = 50; %Time constant (ms) for v1
                        tau2 = 50; %Time constant (ms) for v2
                        tau3 = 20; %Time constant (ms) for u1
                        tau4 = 20; %Time constant (ms) for u2
                        tau_synpla = 30; %Time constant (ms) for synaptic plasticity (depression)
                        
                        %%%%%%%%%%%%  Gain modulation types %%%%%%%%%%%% 
                        if gain == 1 %% Sigmoidal function
                            p1 = 50; 
                            p2 = -0.25;
                            p3 = 70; 
                            p4 = 15; 
                            GM_vec = p1./(1+exp(p2*(-percent_unoccl+p3)))+p4;
                            GM_vec(1) = 1; 
                            if display_fig == 1
                                figure(13),hold on,plot(percent_unoccl,GM_vec)
                            end       
                        elseif gain == 2 %% Step function
                            p1 = 20;
                            GM_vec = p1*ones(1,length(percent_unoccl));
                            GM_vec(1) = 0;
                        elseif gain == 3 %% Linear function
                            a = 0.4;
                            b=40;
                            GM_vec = -a*percent_unoccl+b;  
                        elseif gain == 4 %% Polynomial (cubic) function 
                            p1 =   -0.001743;
                            p2 =      0.3899;
                            p3 =      -29.58;
                            p4 =       806.2;
                            GM_vec = p1*(percent_unoccl).^3 + p2*(percent_unoccl).^2 + p3*(percent_unoccl) + p4 ;
                            if display_fig == 1
                                figure(13),hold on, plot(percent_unoccl,GM_vec,'-o')
                            end
                        elseif gain == 5 %% no gain
                            GM_vec = 30*ones(1,5);   
                        end
                        
                        GM_vec_np =  GM_vec;
                        
                        %%%%%%%%%% Synaptic delays & Threshold %%%%%%%%%%%% 
                        latency = [syn_latency_uv syn_latency_vu 0 0 0]; % synaptic latency (ms), [uv vu vv uu v_self]
                        i_delay = latency./dt;
                        syn_thr = 10;
                        
                        %%%% Solve differntial equations for each occlusion level %%%%%%%%
                        peak1 = zeros(1,length(stim_occl));
                        peak2 = zeros(1,length(stim_occl));
                        
                        frac_correct_peak1 = zeros(1,length(stim_occl));
                        frac_correct_peak2 = zeros(1,length(stim_occl));
                        frac_correct_PFC = zeros(1,length(stim_occl));
                        
                        rocarea_peak1= zeros(1,length(stim_occl));
                        rocarea_peak2= zeros(1,length(stim_occl));
                        rocarea_PFC= zeros(1,length(stim_occl));
                        
                        selectivity_V4_1st = zeros(1,length(stim_occl));
                        selectivity_V4_2nd = zeros(1,length(stim_occl));
                        selectivity_PFC = zeros(1,length(stim_occl));
                        
                        sum_V4_1st = zeros(1,length(stim_occl));
                        sum_V4_2nd = zeros(1,length(stim_occl));
                        sum_PFC = zeros(1,length(stim_occl));
                    
                        v1_sweep_0 = zeros(length(T_grid),length(percent_unoccl));
                        v2_sweep_0 = zeros(length(T_grid),length(percent_unoccl));
                        u1_sweep_0 = zeros(length(T_grid),length(percent_unoccl));
                        u2_sweep_0 = zeros(length(T_grid),length(percent_unoccl));
                        
                        include_population =  zeros(1,length(stim_occl));
                        
                        for i_occl = 1:length(stim_occl)
                            
                            stim_amp = stim_occl(i_occl);
                            stim_amp_np = stim_occl_np(i_occl);
                            GM = GM_vec(i_occl);
                            GM2 = GM_vec_np(i_occl);
                            
                            Input1 = stim_input(T_grid,shape_on,shape_off,stim_amp);
                            Input2 = stim_input(T_grid,shape_on,shape_off,stim_amp_np);
                            
                            ii = 1;
                            t = 0;
                            v1 = zeros(1,length(T_grid));
                            v2 = zeros(1,length(T_grid));
                            u1 = zeros(1,length(T_grid));
                            u2 = zeros(1,length(T_grid));
                            
                            syn_v1 = zeros(1,length(T_grid));
                            syn_v2 = zeros(1,length(T_grid));
                            syn_u1 = zeros(1,length(T_grid));
                            syn_u2 = zeros(1,length(T_grid));
                            
                            f_v1 = zeros(1,length(T_grid));
                            f_v2 = zeros(1,length(T_grid));
                            f_u1 = zeros(1,length(T_grid));
                            f_u2 = zeros(1,length(T_grid));
                            
                            w_inf = [w0_v1u1; w0_v2u2; w0_u1v1; w0_u2v2; w0_u1u2; w0_u2u1; w0_v1v1; w0_v2v2; w0_v1v2; w0_v2v1; w0_v1u2; w0_v2u1; w0_u1v2; w0_u2v1];
                            w_matrix = zeros(14,length(T_grid));
                            w_matrix(:,1) = w_inf;
                            
                            while t < T
                                t = T_grid(ii+1);
                                if whitenoise == 1
                                    noise = sig*randn;
                                else
                                    noise = 0;
                                end
                                
                                if syn_adapt == 1
                                    if u1(ii)>syn_thr
                                        w_inf(1) = 0;
                                        w_inf(12) = 0;
                                    end
                                    
                                    if u2(ii)>syn_thr
                                        w_inf(2) = 0;
                                        w_inf(11) = 0;
                                    end
                                end
                                
                                w_matrix(:,ii+1) = w_matrix(:,ii)+dt*(1/tau_synpla)*(w_inf-w_matrix(:,ii));
                                w_v1u1 = w_matrix(1,ii+1);
                                w_v2u2 = w_matrix(2,ii+1);
                                w_u1v1 = w_matrix(3,ii+1);
                                w_u2v2 = w_matrix(4,ii+1);
                                w_u1u2 = w_matrix(5,ii+1);
                                w_u2u1 = w_matrix(6,ii+1);
                                w_v1v1 = w_matrix(7,ii+1);
                                w_v2v2 = w_matrix(8,ii+1);
                                w_v1v2 = w_matrix(9,ii+1);
                                w_v2v1 = w_matrix(10,ii+1);
                                w_v1u2 = w_matrix(11,ii+1);
                                w_v2u1 = w_matrix(12,ii+1);
                                w_u1v2 = w_matrix(13,ii+1);
                                w_u2v1 = w_matrix(14,ii+1);

                                %%%%%%%%%%%% Synaptic inputs %%%%%%%%%%%%%%%%%%%%%%
                                if ii<= max(i_delay)
                                    syn_v1(ii) = 0;
                                    syn_v2(ii) = 0;
                                    syn_u1(ii) = 0;
                                    syn_u2(ii) = 0;
                                else
                                    
                                    jj_uv = ii-i_delay(1); % PFC to V4 (excitatory)
                                    jj_vu = ii-i_delay(2); % V4 to PFC
                                    jj_vv = ii-i_delay(3); % V4 to V4 (mutual, inhibitory)
                                    jj_uu = ii-i_delay(4); % PFC to PFC
                                    jj_v_self = ii-i_delay(5); % V4 to V4 (recurrent, self excitation)
                                    
                                    if half_wave == 1
                                        PFC_Thr = 30;
                                        PFC_Thr2 = 30;
                                    else
                                        PFC_Thr = 0;
                                        PFC_Thr2 = 0;
                                    end
                                    
                                    syn_v1(ii) = w_u1v1*F(u1(jj_uv),4,0,0,0,PFC_Thr)+w_v1v1*v1(jj_v_self)+w_v2v1*v2(jj_vv) +w_u2v1*F(u2(jj_uv),4,0,0,0,PFC_Thr);
                                    syn_v2(ii) = w_u2v2*F(u2(jj_uv),4,0,0,0,PFC_Thr2)+w_v2v2*v2(jj_v_self)+w_v1v2*v1(jj_vv) +w_u1v2*F(u1(jj_uv),4,0,0,0,PFC_Thr);
                                    syn_u1(ii) = w_v1u1*v1(jj_vu)+w_u2u1*F(u2(jj_uu),4,0,0,0,PFC_Thr2)+w_v2u1*v2(jj_vu);
                                    syn_u2(ii) = w_v2u2*v2(jj_vu)+w_u1u2*F(u1(jj_uu),4,0,0,0,PFC_Thr)+w_v1u2*v1(jj_vu);

                                end
                                
                                
                                %%%%%%%%%%%% Nonlinear filter %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                f_v1(ii) = (1/tau1)*(-v1(ii) + F(syn_v1(ii)-r_th_v1+Input1(ii),NL,sigma1,f_max1,N1,0)+noise);
                                f_v2(ii) = (1/tau2)*(-v2(ii) + F(syn_v2(ii)-r_th_v2+Input2(ii),NL,sigma1, f_max1, N1,0)+noise);
                                
                                if (gain_delay == 1) && (ii<=(50+40+gain_delay_amount)/dt) %%% 50ms until V4 responses start rising,
                                    %%% 40 ms delay from V4 to PFC signal transmission, IT inputs to PFC
                                    %%% starts 10-20 ms after V4 to PFC transmission
                                    f_u1(ii) = (1/tau3)*(-u1(ii) + F((syn_u1(ii)-r_th_u1),NL,sigma2,f_max2,N2,0)+noise); % GM
                                    f_u2(ii) = (1/tau4)*(-u2(ii) + F((syn_u2(ii)-r_th_u2),NL,sigma2,f_max2,N2,0)+noise); %GM
                                else
                                    f_u1(ii) = (1/tau3)*(-u1(ii) + F((syn_u1(ii)-r_th_u1),NL,sigma2,f_max2*GM,N2,0)+noise); % GM
                                    f_u2(ii) = (1/tau4)*(-u2(ii) + F((syn_u2(ii)-r_th_u2),NL,sigma2,f_max2*GM2,N2,0)+noise); %GM
                                end
                                
                                
                                v1(ii+1) = v1(ii)+dt*f_v1(ii); % V4
                                v2(ii+1) = v2(ii)+dt*f_v2(ii); % V4
                                u1(ii+1) = u1(ii)+dt*f_u1(ii); % PFC
                                u2(ii+1) = u2(ii)+dt*f_u2(ii); % PFC
                                
                                ii = ii+1;
                            end
                            
                            %%%%%%%%%%%%%%%%%%%% Plotting Inputs & Responses %%%%%%%%%%%%%%%%%%%%%
                            if display_fig == 1
                                figure(1), set(gcf,'color','w')
                                hold on, subplot(1,2,1), plot(T_grid,Input1,'color',C(i_occl,:),'LineWidth',2),title('Input to V4 Preferred'), xlim([0 700]), ylim([0 200])
                                xlabel('Time (ms)'), ylabel('Visual Input')
                                set(gca,'FontSize',15,'fontWeight','bold'), set(findall(gcf,'type','text'),'FontSize',15,'fontWeight','bold')
                                hold on, subplot(1,2,2), plot(T_grid,Input2,'color',C(i_occl,:),'LineWidth',2),title('Input to V4 Non-preferred'), xlim([0 700]), ylim([0 200])
                                xlabel('Time (ms)'),
                                set(gca,'FontSize',15,'fontWeight','bold'), set(findall(gcf,'type','text'),'FontSize',15,'fontWeight','bold')
                                current_unoccl = percent_unoccl(i_occl);
                                legendInfo{i_occl} = [num2str(current_unoccl),'% unoccluded'];
                                legend(legendInfo)
                                
                                figure(2), set(gcf,'color','w')
                                hold on, subplot(2,2,1), plot(T_grid,u1,'color',C(i_occl,:),'LineWidth',2),title('PFC Preferred (u1)'), xlim([0 700]), ylim([0 70]), box off
                                ylabel('Mean Firing Rate (s^{-1})')
                                set(gca,'FontSize',15,'fontWeight','bold'), set(findall(gcf,'type','text'),'FontSize',15,'fontWeight','bold')
                                hold on, subplot(2,2,2), plot(T_grid,u2,'color',C(i_occl,:),'LineWidth',2),title('PFC Nonpreferred (u2)'), xlim([0 700]), ylim([0 70]), box off
                                set(gca,'FontSize',15,'fontWeight','bold'), set(findall(gcf,'type','text'),'FontSize',15,'fontWeight','bold')
                                hold on, subplot(2,2,3), plot(T_grid,v1,'color',C(i_occl,:),'LineWidth',2),title('V4 Preferred (v1)'), xlim([0 700]), ylim([0 70]), box off
                                ylabel('Mean Firing Rate (s^{-1})')
                                set(gca,'FontSize',15,'fontWeight','bold'), set(findall(gcf,'type','text'),'FontSize',15,'fontWeight','bold')
                                hold on, subplot(2,2,4), plot(T_grid,v2,'color',C(i_occl,:),'LineWidth',2),title('V4 Nonpreferred (v2)'), xlim([0 700]), ylim([0 70]), box off
                                xlabel('Time (ms)'),
                                set(gca,'FontSize',15,'fontWeight','bold'), set(findall(gcf,'type','text'),'FontSize',15,'fontWeight','bold')
                            end
                            
                            
                            %%%%%%5%% Find the averaged responses during the 1st & the 2nd peaks
                            if (whitenoise == 0) && (syn_adapt == 1) %&& (half_wave == 1)
                                
                                window = 30;
                                
                                %% Finding the max/average firing rate of the 1st and the 2nd peaks
                                [xmax,imax,xmin,imin] = extrema(v1);
                                i_1 = find(imax==min(imax));
                                i_2 = find(imax==max(imax));
 
                                peak1_time = T_grid(min(imax));

                                if length(xmax)<2
                                    peak2_time = 200;
                                    xmax(2) = v1(1+peak2_time/dt);
                                    i_2 = 2;
                                    include_population(i_occl) = 0;
                                else
                                    peak2_time = T_grid(imax(i_2));
                                    include_population(i_occl) = 1;
                                end
                                
                                peak1_start = peak1_time-window/2;
                                peak1_end = peak1_time+window/2;
                                peak2_start = peak2_time-window/2;
                                peak2_end = peak2_time+window/2;
                                
                                if length(xmin)<1
                                    trough = 158;
                                    xmin = v1(1+trough/dt);
                                    imin = 1+trough/dt;
                                end
                                
                                i_temp1 = fix(1+peak1_time/dt);
                                i_temp2 = fix(1+peak2_time/dt);
                                
                                if (display_fig == 1) && (fig_peakstar == 1)
                                    figure(2),hold on, subplot(2,2,3), plot(peak1_time,xmax(i_1),'*')
                                    hold on, subplot(2,2,3), plot(peak2_time,xmax(i_2),'*')
                                    hold on, subplot(2,2,3), plot(T_grid(imin),xmin,'*')
                                    
                                    figure(2),hold on, subplot(2,2,4), plot(peak1_time,v2(i_temp1),'*')
                                    hold on, subplot(2,2,4), plot(peak2_time,v2(i_temp2),'*')
                                end

                                selectivity_V4_1st(i_occl) = xmax(i_1)-v2(i_temp1);                               
                                selectivity_V4_2nd(i_occl) =  xmax(i_2)-v2(i_temp2);
                                
                                sum_V4_1st(i_occl) = xmax(i_1)+v2(i_temp1);  
                                sum_V4_2nd(i_occl) = xmax(i_2)+v2(i_temp2);
                                
                                peak1_dur = T_grid(1+peak1_start/dt:1+peak1_end/dt);
                                peak2_dur = T_grid(1+peak2_start/dt:1+peak2_end/dt);
                                
                                %%%%%%%%%%% v1: the preferred V4 population %%%%%%%%%%%
                                v1_peak1_FR = v1(1+peak1_start/dt:1+peak1_end/dt);
                                v1_peak2_FR = v1(1+peak2_start/dt:1+peak2_end/dt);
                                v1_aveFR_peak1 = (1/(peak1_end-peak1_start))*trapz(peak1_dur, v1_peak1_FR);
                                v1_aveFR_peak2 = (1/(peak2_end-peak2_start))*trapz(peak2_dur, v1_peak2_FR);
                                v1_aveFR1(trialnum,i_occl) = v1_aveFR_peak1;
                                v1_aveFR2(trialnum,i_occl) = v1_aveFR_peak2-xmin;

                                %%%%%%%%%%% v2: the non-preferred V4 population %%%%%%%%%%% 
                                v2_peak1_FR = v2(1+peak1_start/dt:1+peak1_end/dt);
                                v2_peak2_FR = v2(1+peak2_start/dt:1+peak2_end/dt);
                                v2_aveFR_peak1 = (1/(peak1_end-peak1_start))*trapz(peak1_dur, v2_peak1_FR);
                                v2_aveFR_peak2 = (1/(peak2_end-peak2_start))*trapz(peak2_dur, v2_peak2_FR);
                                v2_aveFR1(trialnum,i_occl) = v2_aveFR_peak1;
                                v2_aveFR2(trialnum,i_occl) = v2_aveFR_peak2;
 
                                %% Finding the max/average firing rate of the PFC peak
                                [xmax1,imax1,xmin1,imin1] = extrema(u1);
                                ii_1 = imax1==min(imax1);
                                PFC_peak_time1 = T_grid(min(imax1));
                                PFC_peak_start = PFC_peak_time1-window/2;
                                PFC_peak_end = PFC_peak_time1+window/2;
                                
                                if (display_fig == 1) && (fig_peakstar == 1)
                                    figure(2),hold on, subplot(2,2,1), plot(PFC_peak_time1,xmax1(ii_1),'*')
                                end
                                
                                PFC_peak_dur = T_grid(1+PFC_peak_start/dt:1+PFC_peak_end/dt);
                                
                                [xmax2,imax2,xmin2,imin2] = extrema(u2);
                                jj_1 = find(imax2==min(imax2));
                                jj_2 = find(imax2==max(imax2));
                                PFC_peak_time2 = T_grid(min(imax2));
                                
                                if (display_fig == 1) && (fig_peakstar == 1)
                                    figure(2),hold on, subplot(2,2,2), plot(PFC_peak_time2,xmax2(jj_1),'*')
                                end
                                
                                selectivity_PFC(i_occl) = xmax1(1)-xmax2(1);%xmax1(jj_1)-xmax2(jj_1);
                                sum_PFC(i_occl) = xmax1(1) + xmax2(1);
                                
                                %%%%%%%%%%% u1: preferred PFC population %%%%%%%%%%%
                                u1_peak_FR = u1(1+PFC_peak_start/dt:1+PFC_peak_end/dt);
                                u1_aveFR_peak = (1/(PFC_peak_end-PFC_peak_start))*trapz(PFC_peak_dur, u1_peak_FR);
                                u1_aveFR(trialnum,i_occl) = u1_aveFR_peak;

                                %%%%%%%%%%% u2: non-preferred PFC population %%%%%%%%%%%
                                u2_peak_FR = u2(1+PFC_peak_start/dt:1+PFC_peak_end/dt);
                                u2_aveFR_peak = (1/(PFC_peak_end-PFC_peak_start))*trapz(PFC_peak_dur, u2_peak_FR);
                                u2_aveFR(trialnum,i_occl) = u2_aveFR_peak;
                                
                            end
                            
                            %% Poisson spike generator
                            if poisson == 1
                                xvalues = [0:0.5:60];
                                repeat = 1000;
                                %%%%%%%%%%%%%% Peak 1 %%%%%%%%%%%%%%
                                dt_poisson = dt*10;
                                dur_poisson = (peak1_end-peak1_start)*10;
                                spikecounts1 = poissonspike(v1_peak1_FR,dt_poisson,dur_poisson,repeat);
                                spikecounts2 = poissonspike(v2_peak1_FR,dt_poisson,dur_poisson,repeat);
                               
                                [counts1, values1] = hist(spikecounts1,xvalues);
                                [counts2, values2] = hist(spikecounts2,xvalues);
                                
                                rocarea_peak1(i_occl) = roc(spikecounts1, spikecounts2, 1);

                                figure(5), set(gcf,'color','w')
                                subplot(length(stim_occl),1,i_occl)
                                stairs(values1,counts1,'b')
                                hold on, stairs(values2,counts2,'r')
                                
                                str = percent_unoccl(i_occl);
                                title([num2str(str),'% Unoccluded'])
                                if i_occl == 1
                                    legend('Preferred V4 (Peak 1)','Nonpreferred V4  (Peak 1)')
                                elseif i_occl == length(stim_occl)
                                    xlabel('spike counts'), ylabel('# of trials')
                                end
                                
                                minOfHists = min([counts1; counts2], [], 1);
                                overlappedHist = sum(minOfHists);
                                frac_correct_peak1(i_occl) = (1/(2*repeat))*(2*repeat-overlappedHist);
                                
                                %%%%%%%%%%%%%% Peak 2 %%%%%%%%%%%%%%
                                dt_poisson = dt*10;
                                dur_poisson = (peak2_end-peak2_start)*10;
                                spikecounts1 = poissonspike(v1_peak2_FR,dt_poisson,dur_poisson,repeat);
                                spikecounts2 = poissonspike(v2_peak2_FR,dt_poisson,dur_poisson,repeat);
                                
                                [counts1, values1] = hist(spikecounts1,xvalues);
                                [counts2, values2] = hist(spikecounts2,xvalues);
                                
                                rocarea_peak2(i_occl) = roc(spikecounts1, spikecounts2, 1);
                                
                                figure(6), set(gcf,'color','w')
                                subplot(length(stim_occl),1,i_occl)
                                stairs(values1,counts1,'b')
                                hold on, stairs(values2,counts2,'r')
                                
                                str = percent_unoccl(i_occl);
                                title([num2str(str),'% Unoccluded'])
                                if i_occl == 1
                                    legend('Preferred V4 (Peak 2)','Nonpreferred V4  (Peak 2)')
                                elseif i_occl == length(stim_occl)
                                    xlabel('spike counts'), ylabel('# of trials')
                                end
                                
                                minOfHists = min([counts1; counts2], [], 1);
                                overlappedHist = sum(minOfHists);
                                frac_correct_peak2(i_occl) = (1/(2*repeat))*(2*repeat-overlappedHist);
                                
                                
                                %%%%%%%%%%%%%% Poisson spikes from PFC %%%%%%%%%%%%%%
                                dt_poisson = dt*10;
                                dur_poisson = (PFC_peak_end-PFC_peak_start)*10;
                                spikecounts1 = poissonspike(u1_peak_FR,dt_poisson,dur_poisson,repeat);
                                spikecounts2 = poissonspike(u2_peak_FR,dt_poisson,dur_poisson,repeat);
                                
                                [counts1, values1] = hist(spikecounts1,xvalues);
                                [counts2, values2] = hist(spikecounts2,xvalues);
                                
                                rocarea_PFC(i_occl) = roc(spikecounts1, spikecounts2, 1);
                                
                                figure(7), set(gcf,'color','w')
                                subplot(length(stim_occl),1,i_occl)
                                stairs(values1,counts1,'b')
                                hold on, stairs(values2,counts2,'r')
                                
                                str = percent_unoccl(i_occl);
                                title([num2str(str),'% Unoccluded'])
                                if i_occl == 1
                                    legend('Preferred PFC','Nonpreferred PFC')
                                elseif i_occl == length(stim_occl)
                                    xlabel('spike counts'), ylabel('# of trials')
                                end
                                
                                minOfHists = min([counts1; counts2], [], 1);
                                overlappedHist = sum(minOfHists);
                                frac_correct_PFC(i_occl) = (1/(2*repeat))*(2*repeat-overlappedHist);
                                
                                
                            end

                            figure(3), set(gcf,'color','w')
                            hold on, subplot(2,4,1), plot(T_grid,u1,'color',C(i_occl,:),'LineWidth',2),
                            
                            hold on, subplot(2,4,2), plot(T_grid,u2,'color',C(i_occl,:),'LineWidth',2),
                            hold on, subplot(2,4,5), plot(T_grid,v1,'color',C(i_occl,:),'LineWidth',2),
                            
                            hold on, subplot(2,4,6), plot(T_grid,v2,'color',C(i_occl,:),'LineWidth',2),
                            
                            if i_occl==1
                                hold on, subplot(2,4,5), title('V4 Preferred (v1)'), xlim([0 700]), ylim([0 max(max(v1),max(v2))+5]), box off, ylabel('Mean Firing Rate (s^{-1})'), xlabel('Time (ms)')
                                hold on, subplot(2,4,6), title('V4 Nonpreferred (v2)'), xlim([0 700]), ylim([0 max(max(v1),max(v2))+5]), box off, xlabel('Time (ms)')
                            elseif i_occl == length(stim_occl)
                                hold on, subplot(2,4,1), title('PFC Preferred (u1)'), xlim([0 700]), ylim([0 max(max(u1),max(u2))+5]), box off, ylabel('Mean Firing Rate (s^{-1})')
                                hold on, subplot(2,4,2), title('PFC Nonpreferred (u2)'), xlim([0 700]), ylim([0 max(max(u1),max(u2))+5]), box off
                            end
                            set(figure(3), 'Position',[74,79,1395,513] );%[74,79,1228,716]);
                            
                            if (fig_peakstar == 1)
                                if (whitenoise == 0) && (syn_adapt == 1) %&& (half_wave == 1)
                                    figure(3),hold on, subplot(2,4,5), plot(peak1_time,xmax(i_1),'*')
                                    hold on, subplot(2,4,5), plot(peak2_time,xmax(i_2),'*')
                                    hold on, subplot(2,4,5), plot(T_grid(imin),xmin,'*')
                                    hold on, subplot(2,4,6), plot(peak1_time,v2(i_temp1),'*')
                                    hold on, subplot(2,4,6), plot(peak2_time,v2(i_temp2),'*')
                                    hold on, subplot(2,4,1), plot(PFC_peak_time1,xmax1(ii_1),'*')
                                    hold on, subplot(2,4,2), plot(PFC_peak_time2,xmax2(jj_1),'*')
                                end
                            end

                                v1_sweep_0(:,i_occl) = v1;
                                v2_sweep_0(:,i_occl) = v2;
                                u1_sweep_0(:,i_occl) = u1;
                                u2_sweep_0(:,i_occl) = u2;

                            drawnow
                        end
                        

                        
                        if (whitenoise == 0) && (syn_adapt == 1) %&& (half_wave == 1)
                            
                            norm_selectivity = max([selectivity_V4_1st, selectivity_V4_2nd, selectivity_PFC]);
                            
                            figure(3),hold on, subplot(2,4,3), plot(percent_unoccl,v1_aveFR1,'k-o','LineWidth',2)
                            hold on, subplot(2,4,3), plot(percent_unoccl, v1_aveFR2,'m-o','LineWidth',2)
                            xlabel('% Unoccluded'), ylabel('Average response (s^{-1})'),legend('1st peak', '2nd peak', 'Location','Best'),
                            xlim([55 100]), title('Averaged responses')
                            
                            if selectivity_type == 1
                                selectivity_V4_1st_norm = selectivity_V4_1st./norm_selectivity;
                                selectivity_V4_2nd_norm = selectivity_V4_2nd./norm_selectivity;
                                selectivity_PFC_norm = selectivity_PFC./norm_selectivity;
                            elseif selectivity_type == 2
                                selectivity_V4_1st_norm = selectivity_V4_1st./sum_V4_1st;
                                selectivity_V4_2nd_norm = selectivity_V4_2nd./sum_V4_2nd;
                                selectivity_PFC_norm = selectivity_PFC./sum_PFC;
                            end

                            
                            if (max(include_population)>0) 
                                
                                i_sweep  = i_sweep +1;
                                v1_sweep(i_sweep,:,:) = v1_sweep_0;
                                v2_sweep(i_sweep,:,:) = v2_sweep_0;
                                u1_sweep(i_sweep,:,:) = u1_sweep_0;
                                u2_sweep(i_sweep,:,:) = u2_sweep_0;
                                
                                selectivity_V4_1st_sweep(i_sweep) = mean(selectivity_V4_1st(2:end));
                                selectivity_V4_2nd_sweep(i_sweep) = mean(selectivity_V4_2nd(2:end));
                                selectivity_PFC_sweep(i_sweep) = mean(selectivity_PFC(2:end));
                            end

                        end
                        
                        annotation('textbox',...
                            [0 0 0.08 0.17],...
                            'String',{['w_{ff,same} =', num2str(ff_same)...
                            ', w_{ff,cross} =', num2str(ff_cross)...
                            ', w_{fb,same} =', num2str(fb_same),...
                            ', w_{fb,cross} =', num2str(fb_cross)]},...
                            'FontSize',9,...
                            'FontName','Arial',...
                            'LineStyle','none',...
                            'EdgeColor','none',...
                            'LineWidth',1,...
                            'BackgroundColor',[0.9 0.9 0.9]...
                            );
                        
                        drawnow
                    end
                    
                    if poisson == 1
                        %% Fraction correct plot
                        %figure(3), set(gcf,'color','w')
                        %hold on, subplot(2,4,4), plot(percent_unoccl,frac_correct_peak1,'k-o','LineWidth',2)
                        %hold on, subplot(2,4,4), plot(percent_unoccl,frac_correct_peak2,'k:*','LineWidth',2)
                        %hold on, subplot(2,4,4), plot(percent_unoccl, frac_correct_PFC,'b-o','LineWidth',2)
                        %legend('V4 1st peak','V4 2nd peak', 'PFC', 'Location','Best')
                        %xlabel('% Unoccluded'), ylabel('fraction correct')
                        %xlim([55 100]), ylim([0.5 1]), title('Shape selectivity: fraction correct')
                        
                        
                        %% ROC plot
                        figure(3), 
                        hold on, subplot(2,4,7), plot(percent_unoccl,rocarea_peak1,'k-o','LineWidth',2)
                        hold on, subplot(2,4,7), plot(percent_unoccl,rocarea_peak2,'k:*','LineWidth',2)
                        hold on, subplot(2,4,7), plot(percent_unoccl,rocarea_PFC,'b-o','LineWidth',2)
                        legend('V4 1st peak','V4 2nd peak', 'PFC', 'Location','Best')
                        xlabel('% Unoccluded'), ylabel('ROC')
                        xlim([55 100]), ylim([0.5 1]), title('Shape selectivity: ROC')
                        

                    end

                    
                end
            end
        end
    end
end

v1_sweep = v1_sweep(1:i_sweep,:,:);
v2_sweep = v2_sweep(1:i_sweep,:,:);
u1_sweep = u1_sweep(1:i_sweep,:,:);
u2_sweep = u2_sweep(1:i_sweep,:,:);

if selectivity_type == 1
    select_norm = max(selectivity_V4_1st_sweep);
    selectivity_V4_1st_fin = selectivity_V4_1st_sweep./select_norm;
    selectivity_V4_2nd_fin = selectivity_V4_2nd_sweep./select_norm;
    selectivity_V4_PFC_fin = selectivity_PFC_sweep./select_norm;
elseif selectivity_type == 2
    selectivity_V4_1st_fin = selectivity_V4_1st_sweep;
    selectivity_V4_2nd_fin = selectivity_V4_2nd_sweep;
    selectivity_V4_PFC_fin = selectivity_PFC_sweep;
end

%% Averaged responses over parameter sweeps
if fig_type == 2

    figure(10), set(gcf,'color','w')
    scatter(selectivity_V4_1st_fin,selectivity_V4_2nd_fin), hold on, plot([0:0.01:1], [0:0.01:1],'r')

    v1_ave =  squeeze(mean(v1_sweep));
    v2_ave =  squeeze(mean(v2_sweep));
    u1_ave =  squeeze(mean(u1_sweep));
    u2_ave =  squeeze(mean(u2_sweep));
    
    v1_aveFR1 = zeros(1,length(stim_occl));
    v1_aveFR2 = zeros(1,length(stim_occl));
    v2_aveFR1 = zeros(1,length(stim_occl));
    v2_aveFR2 = zeros(1,length(stim_occl));
    
    selectivity_V4_1st = zeros(1,length(stim_occl));
    selectivity_V4_2nd = zeros(1,length(stim_occl));
    selectivity_PFC = zeros(1,length(stim_occl));
    
    sum_V4_1st = zeros(1,length(stim_occl));
    sum_V4_2nd = zeros(1,length(stim_occl));
    sum_PFC = zeros(1,length(stim_occl));
    
    for j_occl = 1:length(stim_occl)
        
        %% Find the averaged responses during the 1st & the 2nd peaks
        if (whitenoise == 0) && (syn_adapt == 1) %&& (half_wave == 1)
            v1_ave_occl = v1_ave(:,j_occl);
            v2_ave_occl = v2_ave(:,j_occl);
            u1_ave_occl = u1_ave(:,j_occl);
            u2_ave_occl = u2_ave(:,j_occl);
            %% Finding the max/average firing rate of the 1st and the 2nd peaks
            [xmax,imax,xmin,imin] = extrema(v1_ave_occl);
            i_1 = find(imax==min(imax));
            i_2 = find(imax==max(imax));
            peak1_time_ave = T_grid(min(imax));
            
            if length(xmax)<2
                peak2_time_ave = 200;
                xmax(2) = v1_ave_occl(1+peak2_time_ave/dt);
                i_2 = 2;
            else
                peak2_time_ave = T_grid(imax(i_2));
            end
            
            peak1_start_ave = peak1_time_ave-window/2;
            peak1_end_ave = peak1_time_ave+window/2;
            peak2_start_ave = peak2_time_ave-window/2;
            peak2_end_ave = peak2_time_ave+window/2;
            
            if length(xmin)<1
                trough = 158;
                xmin = v1_ave_occl(1+trough/dt);
                imin = 1+trough/dt;
            elseif length(xmin)>1
                xmin = xmin(1);
                imin = imin(1);
            end
            
            selectivity_V4_1st(j_occl) = xmax(i_1)-v2_ave_occl(1+peak1_time_ave/dt);
            selectivity_V4_2nd(j_occl) =  xmax(i_2)-v2_ave_occl(1+peak2_time_ave/dt);
            
            if selectivity_type == 2
                sum_V4_1st(j_occl)= xmax(i_1)+v2_ave_occl(1+peak1_time_ave/dt);
                sume_V4_2nd(j_occl)=  xmax(i_2)+v2_ave_occl(1+peak2_time_ave/dt);
            end
            
            peak1_dur_ave = T_grid(1+peak1_start_ave/dt:1+peak1_end_ave/dt);
            peak2_dur_ave = T_grid(1+peak2_start_ave/dt:1+peak2_end_ave/dt);
            
            
            %% v1: the preferred V4 population
            v1_peak1_FR_ave = v1_ave_occl(1+peak1_start_ave/dt:1+peak1_end_ave/dt);
            v1_peak2_FR_ave = v1_ave_occl(1+peak2_start_ave/dt:1+peak2_end_ave/dt);
            
            v1_aveFR_peak1_ave = (1/(peak1_end_ave-peak1_start_ave))*trapz(peak1_dur_ave, v1_peak1_FR_ave);
            v1_aveFR_peak2_ave = (1/(peak2_end_ave-peak2_start_ave))*trapz(peak2_dur_ave, v1_peak2_FR_ave);
            
            v1_aveFR1(j_occl) = v1_aveFR_peak1_ave;
            v1_aveFR2(j_occl) = v1_aveFR_peak2_ave-xmin;
            
            %% v2: the non-preferred V4 population
            v2_peak1_FR_ave = v2_ave_occl(1+peak1_start_ave/dt:1+peak1_end_ave/dt);
            v2_peak2_FR_ave = v2_ave_occl(1+peak2_start_ave/dt:1+peak2_end_ave/dt);
            v2_aveFR_peak1_ave = (1/(peak1_end_ave-peak1_start_ave))*trapz(peak1_dur_ave, v2_peak1_FR_ave);
            v2_aveFR_peak2_ave = (1/(peak2_end_ave-peak2_start_ave))*trapz(peak2_dur_ave, v2_peak2_FR_ave);
            
            v2_aveFR1(j_occl) = v2_aveFR_peak1_ave;
            v2_aveFR2(j_occl) = v2_aveFR_peak2_ave;
            
            
            %% Finding the max/average firing rate of the PFC peak
            [xmax1,imax1,xmin1,imin1] = extrema(u1_ave_occl);
            ii_1 = imax1==min(imax1);
            PFC_peak_time_ave = T_grid(min(imax1));
            
            PFC_peak_start_ave = PFC_peak_time_ave-window/2;
            PFC_peak_end_ave = PFC_peak_time_ave+window/2;
            
            PFC_peak_dur_ave = T_grid(1+PFC_peak_start_ave/dt:1+PFC_peak_end_ave/dt);
            
            [xmax2,imax2,xmin2,imin2] = extrema(u2_ave_occl);
            jj_1 = find(imax2==min(imax2));
            jj_2 = find(imax2==max(imax2));
            PFC_peak_time_ave = T_grid(min(imax2));
            
            selectivity_PFC(j_occl) = abs(xmax1(jj_1)-xmax2(jj_1));
            sum_PFC(j_occl) = abs(xmax1(jj_1)+xmax2(jj_1));
            
            %% u1: preferred PFC population
            u1_peak_FR_ave = u1_ave_occl(1+PFC_peak_start_ave/dt:1+PFC_peak_end_ave/dt);
            u1_aveFR_peak_ave = (1/(PFC_peak_end_ave-PFC_peak_start_ave))*trapz(PFC_peak_dur_ave, u1_peak_FR_ave);
            
            u1_aveFR(j_occl) = u1_aveFR_peak_ave;
            
            
            %% u2: non-preferred PFC population
            u2_peak_FR_ave = u2_ave_occl(1+PFC_peak_start_ave/dt:1+PFC_peak_end_ave/dt);
            u2_aveFR_peak_ave = (1/(PFC_peak_end_ave-PFC_peak_start_ave))*trapz(PFC_peak_dur_ave, u2_peak_FR_ave);
            
            u2_aveFR(j_occl) = u2_aveFR_peak_ave;
            
            norm_selectivity = max([selectivity_V4_1st, selectivity_V4_2nd, selectivity_PFC]);
            
        end
        
        %% Population Figures
        
        figure(8), set(gcf,'color','w')
        hold on, subplot(2,3,1), plot(T_grid,u1_ave(:,j_occl),'color',C(j_occl,:),'LineWidth',2),
        title('Averaged PFC Preferred (u1)'), xlim([0 700]), %ylim([0 max(max(u1_ave),max(u2_ave))+5])
        box off, ylabel('Mean Firing Rate (s^{-1})')
        hold on, subplot(2,3,2), plot(T_grid,u2_ave(:,j_occl),'color',C(j_occl,:),'LineWidth',2),
        title('Averaged PFC Nonpreferred (u2)'), xlim([0 700]), %ylim([0 max(max(u1_ave),max(u2_ave))+5]),
        box off
        hold on, subplot(2,3,4), plot(T_grid,v1_ave(:,j_occl),'color',C(j_occl,:),'LineWidth',2),
        title('Averaged V4 Preferred (v1)'), xlim([0 700]), %ylim([0 max(max(v1_ave),max(v2_ave))+5]),
        box off, ylabel('Mean Firing Rate (s^{-1})'), xlabel('Time (ms)')
        hold on, subplot(2,3,5), plot(T_grid,v2_ave(:,j_occl),'color',C(j_occl,:),'LineWidth',2),
        title('Averaged V4 Nonpreferred (v2)'), xlim([0 700]),%ylim([0 max(max(v1_ave),max(v2_ave))+5]),
        box off, xlabel('Time (ms)')
        
        
        if fig_peakstar == 1
            figure(8),hold on, subplot(2,3,4), plot(peak1_time_ave,xmax(i_1),'*')
            hold on, subplot(2,3,4), plot(peak2_time_ave,xmax(i_2),'*')
            hold on, subplot(2,3,4), plot(T_grid(imin),xmin,'*')
            
            figure(8),hold on, subplot(2,3,5), plot(peak1_time_ave,v2_ave_occl(1+peak1_time_ave/dt),'*')
            hold on, subplot(2,3,5), plot(peak2_time_ave,v2_ave_occl(1+peak2_time_ave/dt),'*')
            
        end
    end
    
    figure(8),hold on, subplot(2,3,3), plot(percent_unoccl,v1_aveFR1,'k-o','LineWidth',2)
    hold on, subplot(2,3,3), plot(percent_unoccl, v1_aveFR2,'m-o','LineWidth',2)
    xlabel('% Unoccluded'), ylabel('Average response (s^{-1})'),legend('1st peak', '2nd peak'),
    title('Averaged responses')
    
    if selectivity_type == 1
        figure(8), hold on, subplot(2,3,6), plot(percent_unoccl,selectivity_V4_1st./norm_selectivity ,'k-o','LineWidth',2)
        hold on, subplot(2,3,6), plot(percent_unoccl, selectivity_V4_2nd./norm_selectivity ,'k:*','LineWidth',2)
        hold on, subplot(2,3,6), plot(percent_unoccl, selectivity_PFC./norm_selectivity ,'b-o','LineWidth',2)
    elseif selectivity_type == 2
        figure(8), hold on, subplot(2,3,6), plot(percent_unoccl,selectivity_V4_1st./sum_V4_1st ,'k-o','LineWidth',2)
        hold on, subplot(2,3,6), plot(percent_unoccl, selectivity_V4_2nd./sum_V2_2nd ,'k:*','LineWidth',2)
        hold on, subplot(2,3,6), plot(percent_unoccl, selectivity_PFC./sum_PFC ,'b-o','LineWidth',2)
        
    end
    
    
    legend('V4 selectivity, 1st peak','V4 selectivity, 2nd peak', 'PFC selectivity')
    xlabel('% Unoccluded'), ylabel('Normalized peak response difference')
    title('Shape selectivity (difference between preferred & nonpreferred')
    
    set(figure(8), 'Position',[74,79,1395,513] );
    
    
    %% Gain function Plot
    figure(11), set(gcf,'color','w')
    plot(percent_unoccl, GM_vec,'k-o'),xlabel('% Unoccluded'), ylabel('alpha'), title('Gain function')

    
end

