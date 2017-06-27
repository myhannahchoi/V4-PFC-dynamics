# V4-PFC-dynamics

A rate-based model capturing dynamics of a simplified V4-vlPFC neuronal network. 
Written by Hannah Choi (hannahch@uw.edu), 6/27/2017.

These MATLAB codes generate dynamic responses (firing rates) of V4 (2)  and vlPFC (2) units to simulated preferred/non-preferred shape stimuli under different levels of occlusion. Each of V4 and vlPFC units has shape preference. V4 units receive feedforward sensory stimuli inputs and excitatory feedback inputs from vlPFC units, and vlPFC units receive excitatory feedforward inputs from V4 units. 
1. "Run_dynamics.m" is the main figure generating code that solves the firing rate model by Forward Euler Method. 
2. "stim_input.m" simulates input stimuli to V4 units 1 and 2.  
3.  "F.m" is a function file that calls various forms of nonlinearity used in the model. 
4. "poissonspike.m" generates Poisson spikes with the firing rates obtained in the model.
5.  "roc.m" is called to calculate selectivity by computing the area under the ROC curve, adapted from Anitha Pasupathy. 
6.  "extrema.m" is an extrema finding code, slightly modified from an existing code written by Carlos Adrián Vargas Aguilera and obtained from http://www.mathworks.com/matlabcentral/fileexchange

 Any comments/bug-reports are welcome (hannahch@uw.edu).  

Thank you.
