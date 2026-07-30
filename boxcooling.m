% ===
% Preliminary analysis on air flow requirements and temperature results.
% Massive assumptions made. Must verify via simulation, then preferably
% correlated too.
% Assumes:
% - Constant heat flux -- determined by average packheat from lapheat
% - Perfectly unmixed airflow between rows
% - Straight channels (no curvature of cells)
% - Equal flow on all rows (can add distrubtion later)
% - Boundary layer affects captured by Nusselt number correlations
% Process
% - For each velocity to test
% - For each row
% - Calculate temperature of air at column (calorimetry)
% - Calculate cell temp via transfer coefficient via Nusselt's
% ===

clear; clc; clf; close all;

% parameters
packheatmean = 925; 
cellheat = packheatmean / 600;

% geometric parameters
channelw = 0.065;
channelh = 0.006;
channelh_edge = 0.003; % row 1 & 7's height may be different
channell = 20 * (0.018 + 0.003); % cell diameter plus cell gap-x

% physical properties
c_air = 1006; % J/kgK
rho_air = 1.225; %kg/m3

ms_to_cfm = @(v) channelw * (channelh * 5 + channelh_edge * 2) * v / 0.0283 * 60;

function D=fD(W, H)
    D = 2 * W * H / (W + H);
end

function Re=Re(v, W, H)
    u_air = 1.821 * 1e-5; % viscosity
    D = fD(W,H);
    Re = v * D / u_air;
end

function Nu=Nu(Re) 
    Pr = 0.71; % air. stays constant till 100C (google)

    if Re > 10000
        Nu = 0.023 * Re ^ (4/5) * Pr ^ (0.4);
    else
        Nu = 3.66;
    end
end

function h_conv=h_conv(v, W, H)
    k_air = 0.025; % thermal conductivity.
    D = fD(W,H);
    h_conv = Nu(Re(v, W, H)) * k_air / D;
end


% velocities to plot
v_trials = [3 4 5 6];

for v = v_trials
    % outputs
    re = zeros(1, 7);
    h = zeros(1, 7);
    Tair = zeros(20, 7);
    Tcell = zeros(20, 7);
    Tair(1, :) = 30;

    for row = 1:7
        % rows 1 and 7 are boundaries. only half the thermal load.
        if row == 1 || row == 7
            q = cellheat/2;
        else
            q = cellheat;
        end

        C = c_air * v * channelh * channelw * rho_air;
        re(row) = Re(v, channelw, channelh);
        h(row) = h_conv(Nu(re(row)), channelw, channelh);
        
        for col = 1:20
            % propogate air temp
            Tair(col+1, row) = Tair(col, row) + q / C;
        end
    end

    for row = 1:6
        h_total = h(row) + h(row+1);

        for col = 1:20
            dT = cellheat / (h_total * channell * channelw);
            Tcell(col, row) = Tair(col, row) + dT;
        end
    end

    fprintf("== using velocity: %.2f ==\n", v);
    disp("got Re: ");
    disp(re);

    figure;
    hold on;
    grid on;
    plot(1:20, Tair(1:20, 2), 'x-');
    plot(1:20, Tcell(:, 2), 'o-');
    ylim([25, 60]);
    ylim([25, 60]);
    legend("air temp", "cell temp");
    xlabel("column #");
    ylabel("temperature (C)");
    title(sprintf("flow: %.2f (ms), %.0f (cfm)", v, ms_to_cfm(v)));
end
