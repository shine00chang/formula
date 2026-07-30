% ===
% Interpolates EMRAX 228 efficiency graph
% data from: https://emrax.com/wp-content/uploads/2025/03/EMRAX_228_datasheet_v1.6.pdf
% ===
% Developer's Note: rpm is scaled down by 0.01 because the axis magnitudes
% should be similar for interpolation to look nice.

function F=motor_efficiency()
    % x is rpm, y is torque
    rpm150 = [400 700 1000 1200 4100 5000 ];
    e150   = [.88 .92 .94  .95  .95  .94  ];
    rpm100 = [300 600 950 1400 1600 3500 4000 4500 4750 5000];
    e100   = [.86 .92 .94  .95  .96  .96  .95  .94  .92 .88];
    rpm75 = [300 600 950 1700 2200 3200 4200 4500 4750];
    e75   = [.86 .92 .94  .95  .96  .95  .94  .92  .86];
    rpm50 = [300 1000 3800 4300];
    e50   = [.86 .92   .92  .86];
    rpm25 = [1000 2000 3000 4000 5000];
    e25   = [ .86  .86  .86  .86  .86];
    
    x = [rpm25, rpm50, rpm75, rpm100, rpm150] * 0.01;
    y = [ones(1, length(e25)) 50 * ones(1, length(e50)) 75 * ones(1, length(e75)) 100 * ones(1, length(e100)) 150 * ones(1, length(e150))];
    v = [e25, e50, e75,  e100, e150];
    f = scatteredInterpolant(x.',y.',v.');
    f.Method = "linear";

    F = @(rpm, torque) f(rpm/100, torque);
end

% call this to verify interpolated graph
function plot()
    F = motor_efficiency();

    figure;
    a = 1:50:5000;
    b = 1:5:150;
    [X, Y] = meshgrid(a, b);
    Z = F(X, Y);
    surf(X, Y, Z);
end
            