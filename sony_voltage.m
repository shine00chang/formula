% ===
% For Sony US18650VTC6
% Provides an anonymous function interpolating voltage over Ah
% Data from: https://lygte-info.dk/review/batteries2012/Sony%20US18650VTC6%203000mAh%20%28Green%29%20UK.html
% Using "Discharge 30A" graph (worst-case graph)
% ===

function F=sony_voltage()
    xp = [0  0.25  0.5 0.75    1 1.25 1.5  1.75 2.5];
    yp = [3.6 3.4 3.34 3.28 3.21 3.19 3.15 3.1  2.8];

    F = @(Ah_used) interp1(xp, yp, Ah_used, "linear");

    x = 0:0.1:3;
    y = F(x);
    
    figure;
    plot(x, y);
    title("sony vtc6 voltage")
end