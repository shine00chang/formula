% ===
% For Sony US18650VTC6
% Provides an anonymous function interpolating voltage over Ah
% Data from: https://lygte-info.dk/review/batteries2012/Sony%20US18650VTC6%203000mAh%20%28Green%29%20UK.html
% Using "Discharge 0.2A" graph (closest to true V over SoC)
% ===

function F=sony_voltage()
    xp = [0  0.25  0.5 0.75    1 1.25 1.5  1.75 2 2.25 2.5];
    yp = [4.17 4.08 4.05 3.95 3.86 3.8 3.74 3.66 3.56 3.48 3.32];

    F = @(Ah_used) interp1(xp, yp, Ah_used, "linear");

    x = 0:0.1:3;
    y = F(x);
    
    figure;
    plot(x, y);
    title("sony vtc6 voltage")
end