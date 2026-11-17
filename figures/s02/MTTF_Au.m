% Cross section = 2umx1um, current = 40mA, 
% J = 2MA/cm2, T = ~300C
% MTTF~100hr 
theta = 10000;
LW = (5:1000); % to be converted to (/100nm)
% T = 600; % LW=5, J=2MA/cm2, 6.95 ~ 100hr; J = I/DW =
% 10e-6/100e-9/(LW100e-9)
T=300;
A = 100/6.95*(T/600)^2;
MTTF = (LW/100).^2*exp(0.41/8.617e-5/T) *A;
MTTFactherm = (LW/100).^2.*exp(0.41/8.617e-5./(T+theta*(LW/100).^(-2))) *A;
MTTFAC = MTTF.^2*2*100;
MTTFrec = (LW/100).^2*exp(0.41/8.617e-5/T) *A; % 
figure;
plot(LW,MTTF)
xlabel("linewidth / nm")
ylabel("MTTF / hr")
title(['AC MTTF - linewidth for T=' num2str(T) ])