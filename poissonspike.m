function  [counts] = poissonspike(spikesPerS,timeStep_ms, duration_ms,repeat)
timeStepS = timeStep_ms/1000;
durationS = duration_ms/1000;

spikes = makeSpikes(timeStepS, spikesPerS, durationS, repeat);
rasterPlot(spikes, timeStepS);

counts = countSpikes(spikes,timeStepS);
% plotSpikesCounts(counts);

end

function counts = countSpikes(spikes, timeStepS,startS,endS)
if (nargin < 4)
    endS = length(spikes)*timeStepS;
end
if (nargin < 3) 
    startS = 0;
end
trains = size(spikes,1);
counts = zeros(1,trains);
startBin = startS/timeStepS + 1;
endBin = floor(endS/timeStepS);

for train = 1:trains
    counts(train) = sum(spikes(train,startBin:endBin));
end    
end


function rasterPlot(spikes, timeStepS)
figure(21),clf,set(gcf,'color','w')

times =[0:timeStepS:timeStepS*(length(spikes)-1)];
axes('position',[0.1,0.1,0.8,0.8]);
axis([0, length(spikes)-1,0,1]);
trains = size(spikes, 1);
ticMargin = 0.01;
ticHeight = (1 - (trains + 1) * ticMargin)/ trains;

for train = 1:trains
    spikeTimes = find(spikes(train,:) == 1);
    
    yOffset = ticMargin + (train - 1) * (ticMargin + ticHeight);
    for i = 1:length(spikeTimes)
        line([spikeTimes(i), spikeTimes(i)],[yOffset, yOffset+ticHeight]);
    end
end

xlabel('Time (s)')
title('Raster plot of spikes')

end

function spikes = makeSpikes(timeStepS, spikesPerS, durationS, numTrains)
if (nargin<4)
    numTrains = 1;
end
times = [0:timeStepS:durationS];
spikes = zeros(numTrains, length(times));

for train = 1:numTrains
    vt = rand(size(times));
%     size(spikesPerS)
%     size(timeStepS)
%     size(spikes)
%     size(times)
%     timeStepS
%     durationS
    spikes(train,:) = (spikesPerS.*timeStepS)>vt;
end
end
