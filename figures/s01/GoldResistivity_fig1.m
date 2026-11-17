% https://www.researchgate.net/publication/320255324_Electrical_Resistivity_of_Gold_Thin_Film
%%
close all
clear
thickness=100e-9; % 100nm
% L1=5e-2; % unit: m; 
L2=[5 10 15 20]*1e-3; % m, 5mm-20mm
% S1=thickness *5e-6; % width=5um
widths=(50:1000); % nm

ro_bulk=2.88e-8;
l = 37.7; % nm, mean free path
eta = 12.9; %nm
c = 1.09;
rho0 = 2.2e-8; % ohm m
R = zeros(length(L2),length(widths));
figure;
for i=1:length(L2)

    t = widths;
    t_eff = t-eta;
    k = min(t_eff/2/l,1); % dimensionless: t_eff / l unit = nm
    rho = c*rho0./k./(1-log(k));
%     R1=ro_bulk*L1/S1;

    %ro(i)=(2.88+13.53*exp(-1*width(w)./31.25e-9))/1e8;

    R2=rho.*(L2(i)./(widths/1e9)/thickness);
    i
    plot(widths,rho); hold on;
    R(i,:)=R2;
end

figure;
% a=[width'*1e9,R];
for i=1:length(L2)

    plot(widths,R(i,:)/1000,'LineWidth',2); hold on;
end
% plot(width,ro)
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
% set(gca, 'XDir','reverse')
% o=[width',R];

ythres = 15*ones(1,length(widths)); % plot the dashed line for threshold
plot(widths,ythres,"--","LineWidth",1,"Color",'k');hold on;

xlabel("Linewidth (nm)");
ylabel("Resistance (kΩ)")
xlim([50 1000])
ylim([1 200])
xticks([50 100 200 500 1000])
yticks([1 2 5 10 15 20 50 100 200])
lgd = legend(["5mm";"10mm";"15mm";"20mm"]); %,'location','northeast');
lgd.Color = 'none'; % Makes the legend background transparent
lgd.Box = 'off'; % Turns off the box around the legend

% title('Resistance vs Linewidth ');
%%

