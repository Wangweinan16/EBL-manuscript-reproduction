close all
LW = 5:1000;
T = 300; %K
k = 1.380649e-23; % J/K
C = 5.9e28; % C - Gold atomic concentration: 5.90×10^22 cm-3
D = 1e-20/exp(-0.41/8.617e-5/573)*exp(-0.41/8.617e-5/T); % D: diffusivity ~1e-20m2/s @300C
Omega = 1.25e-29; % Au atomic volume Omega = 1.25 × 10–23 cm3
l = 10e-3; % 10mm
Z = 11; % effective charge number
e = 1e-19; % Coloumb
rho = 2.6e-8; % ohm m
delta = 0.2e-9;
Ev = 1.6e-19;

I=[1,5,10,20]*1e-6;%stimulation inject current, unit is A
figure;
for i=1:length(I)
    j = I(i)./(100*1e-9*LW*1e-9); %J: 10uA/(100nmxLW) % = 100/LW (A/m2)
    MTTF = delta * Ev ./ (Omega * l * (C*D/k/T)*(Z*e*rho.*j).^2); %(sec)
    MTTFh = MTTF/3600;
    MTTFyr = MTTFh/24/365;
    plot(LW,MTTFh,'LineWidth',2);hold on;
end
ythres = 24000*ones(1,length(LW)); % plot the dashed line for threshold
plot(LW,ythres,"--","LineWidth",1,"Color",'k');hold on;

set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
set(gca, 'XDir')
xlabel("Linewidth (nm)")
ylabel("MTTF (hr)")
% Set the specific tick values for the x and y axes
ax = gca; % Get current axes

xTicks = [5 10 20 50 100 200 500 1000];
% yTicks = [1e4 2.4e4 1e6 1e8 1e10];
yTicks = [1e3 2.4e4 1e5 1e7 1e9 1e11];

ax.XTick = xTicks;
ax.YTick = yTicks;

% Set the labels for the specific tick values
% ax.YTickLabel = arrayfun(@(val) sprintf('y = %.2f', val), yTicks, 'UniformOutput', false);
% Custom labels for the specific tick values
% yTickLabels = {'10^4', '24000', '10^6', '10^8', '10^{10}'}; % Custom labels for y-axis
yTickLabels = {'10^3', '24000', '10^5', '10^7', '10^{9}', '10^{11}'}; % Custom labels for y-axis


% Set the custom labels for the specific tick values
ax.YTickLabel = yTickLabels;

% xticks([5 10 20 50 100 200 500 1000])
% yticks([1e4 2.4e4 1e6 1e8 1e10])
% ytickformat('%.1e')
ylim([5e2 1e12]);
% title('MTTF vs linewidth')
lgd=legend(["1uA";"5uA";"10uA";"20uA"],'location','northeast');
lgd.Color = 'none'; % Makes the legend background transparent
lgd.Box = 'off'; % Turns off the box around the legend
