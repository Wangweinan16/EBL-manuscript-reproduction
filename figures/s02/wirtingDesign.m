clear
close
chan_num = 100;
L2=[10]*1e-3; % length of wire in m

current_list=[100,  50,  10,  4 ]*1e-9;
% current_list=[100,   10,  4 ]*1e-9;

dose=250e-6*10000;% C/m2 ~500uC/cm2

width=1000e-9:-1e-9:50e-9;
for i = 1:length(width)

    if width(i)>500e-9
        current=current_list(1);
        j=1;
    elseif width(i)>250e-9
    current=current_list(2);
    j=2;
    elseif width(i)>100e-9
    current=current_list(3);
    j=3;
    elseif width(i)>50e-9
    current=current_list(4);
    j=4;
%         elseif width>20e-9
%     current=current_list(5);
%     j=5;
%     elseif width>10e-9
%     current=current_list(6);
%     j=6;
    end
    
    WT(j,i)=dose.*(L2*width(i))*chan_num/current;
end
c = turbo(length(current_list));
figure;
for j = 1:length(current_list)
% plot(1e9*width(WT(j,:)~=0),WT(j,WT(j,:)~=0),"LineWidth",2,"Color",c(j,:));hold on;
plot(1e9*width(WT(j,:)~=0),WT(j,WT(j,:)~=0),"LineWidth",2);hold on;
end

ythres = 50*ones(1,length(width)); % plot the dashed line for threshold
plot(1e9*width,ythres,"--","LineWidth",1,"Color",'k');hold on;


% xlim([10e0,10e2]);
ylim([10,100]);
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
% set(gca, 'XDir','reverse')
xlabel('Linewidth (nm)');
ylabel('Writing time (s) / 100 channels');
xticks([50 100 200 500 1000])
yticks([20 30 40 50 70 100 120])

% legend(["10nA","5nA","2nA","1nA","0.2nA","0.1nA"],'Location','northwest')
legend(["100nA";"50nA";"10nA";"4nA"],'location','northeast')
legend('boxoff');
% legend(["100nA","10nA","4nA"],'Location','northeast')
% title("Dose time vs Linewidth");
a=[width'*1e9,WT'];