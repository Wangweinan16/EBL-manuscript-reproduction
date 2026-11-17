
clear

width = 100e-9: 10e-9: 1000e-9; % unit is m

% T = 600; % LW=5, J=2MA/cm2, 6.95 ~ 100hr
T=600;%temperature in K 
A=100*4/(exp(0.41/(8.61*6*1e-3)))%constant, unit is h*mA2/cm4
A=A/(3.6*10^5) %constant, unit is s*A2/m4
thickness=100e-9
current=[1,5,10,20]*1e-6%stimulation inject current, unit is A 
R = zeros(length(current),length(100e-9:10e-9:1000e-9));
for i=1:length(current)

    for w=1:length(width)
        j=current(i)/(thickness*width(w));%unit is A/m2
        TTFDC=A*(310./j).^2*exp(0.41/(8.61e-5*310));%unit is second
        TTFAC=2*100.*TTFDC.^2; %unit is second
        R(i,w)=TTFAC;
    end
end
%https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=491258

figure;
for i=1:length(current)
    plot(width*1e9,R(i,:),'LineWidth',2);hold on;
end 

set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
set(gca, 'XDir')


%MTTF = (LW/100).^2*exp(0.41/8.617e-5/T) *A;

%figure;
%plot(LW,MTTF)
xlabel("linewidth / nm")
ylabel("MTTF / s")
title('MTTF vs linewidth')
lgd=legend(["1nA";"5nA";"10nA";"20nA"],'location','northeast');
lgd.Color = 'none'; % Makes the legend background transparent
lgd.Box = 'off'; % Turns off the box around the legend