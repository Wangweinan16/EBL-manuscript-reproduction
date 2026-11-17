LW = [80 100 120];
dTime = [844.41/1.5 867.34/1.3 763.12/1.1]; % dose time
wTime = [1251.28 1367.29 1400.62]; % wait time +data transfer
sTime = 558; % stage time
totTime = dTime+wTime+sTime;
figure;
plot(LW, totTime); hold on;
plot(LW,dTime);
plot(LW,wTime)
xlabel('Linewidth / nm');
ylabel('Writing time / s');
legend(["total time";"dose time";"wait time"]);

LW1 = [80 100 120];% 140 160]
w1Time = [1251.28 1367.29 1400.62];% 1420 1435]; % wait time +data transfer
% myfittype = fittype('a + b*log(1+c*x)','dependent',{'y'},'independent',{'x'},'coefficients',{'a','b','c'})
myfittype = fittype('smoothingspline')

myfit = fit(LW1',w1Time',myfittype)
figure;
plot(myfit,LW1,w1Time)
%%

ddtime = 512*5*10^-3*LW*400*10^-6/(100^2)/(4*10^-9)

myfittype = fittype('smoothingspline')
myfit1 = fit(LW',dTime',myfittype)
figure;
plot(myfit1,LW,dTime)