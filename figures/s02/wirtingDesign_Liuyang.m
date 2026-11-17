clear
L2=[5]*1e-3;

current_list=[10,  5,  2,  1,  0.2,  0.1 ]*1e-9;

dose=500e-6*10000;%C/m2

i=1;
for width=500e-9:-1e-9:5e-9
    if width>300e-9
        current=current_list(1);
        
    elseif width>200e-9
    current=current_list(2);
    
    elseif width>100e-9
    current=current_list(3);
    
    elseif width>50e-9
    current=current_list(4);
    
        elseif width>20e-9
    current=current_list(5);
    
    elseif width>10e-9
    current=current_list(6);
    end
    
    WT(i)=dose.*(L2*width)*128/current;
i=i+1
end
width=500e-9:-1e-9:5e-9;
plot(width,WT)    

a=[width'*1e9,WT'];