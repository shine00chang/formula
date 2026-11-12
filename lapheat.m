% ===
% This script uses Assetto Corsa Simulation Driver input & engine data to 
% determine powertrain thermal losses over one lap
% ===

clear;

filename = "datasets/mk11_150Nm_30NmRegen_STANDARDPACE.csv";
% filename = "datasets/mk11_150Nm_30NmRegen_FULLCONSERVE.csv";
% filename = "datasets/mk11_150Nm_30NmRegen_FULLPUSH.csv";
% filename = "datasets/mk11_150Nm_30NmRegen_SEMIPUSH.csv";
% first three rows are empty
raw = readmatrix(filename);
%raw = readmatrix("datasets/lapheat_autox_loop_1_lap_116Nm.csv");
%raw = readmatrix("datasets/lapheat_autox_MIS_1_lap_80kW.csv");

% extract
t = raw(:, 1);
rpm = raw(:,2);
torque = raw(:,3);
throttle = raw(:,5) / 100;

% grab interpolations
motor_efficiency=motor_efficiency();
sony_voltage=sony_voltage();

% properties
drivetrainefficiency = 0.928;
FDR = 4.91;
wheelR = 0.2;

packconfig = [114, 6];
cellIR = 0.017; % mOhm
bondIR = 2 * 0.001; 
packIR = (cellIR + bondIR) * packconfig(1) / packconfig(2) + 0.050; % +50mOhm for buses and HV components.

% longitudinal acceleration
dw = diff(movmean(rpm, 10) * 2 * pi / 60) / mean(diff(t));
dw = [0; dw];
accel_long = dw / FDR * wheelR / 9.8;

% these refer to the power going OUT of the components they are named.
wheelpower = torque .* rpm * 2 * pi / 60;
motorpower = wheelpower / drivetrainefficiency;

% motor
motoreff = arrayfun(@(a, b) motor_efficiency(a,b), rpm, torque);
motorheat = abs(motorpower) .* (1-motoreff);
motorheatmean = mean(motorheat);
inverterpower = motorpower ./ motoreff;

% pack
% pack power cannot account for inverter losses because inverter losses are
% correlated to input voltage and current. chicken and egg problem. see
% note in inverter section.
packpower = inverterpower;
packpowermean = mean(packpower);
packvoltage = zeros(1, length(t));
packcurrent = zeros(1, length(t));
packvoltage(1) = 420;

Ah_used = 0;
energy_used = 0; % Joules
regen_energy = 0;
for i = 2:length(t)
    cellv = sony_voltage(Ah_used / packconfig(2));
    packv = cellv * packconfig(1);
    
    % solve for current accounting for voltage sag
    if packpower(i) > 0
        r = roots([-packIR, packv, -packpower(i)]);
        packcurrent(i) = min(r);
    else 
        r = roots([packIR, packv, -packpower(i)]);
        packcurrent(i) = max(r);
    end

    % calculate voltage
    if packcurrent(i) == 0
        packvoltage(i) = packv;
    else 
        packvoltage(i) = abs(packpower(i) / packcurrent(i));
    end

    Ah_used = Ah_used + packcurrent(i) * (t(i) - t(i-1)) / 3600;
    energy_used = energy_used + packvoltage(i) * packcurrent(i) * (t(i) - t(i-1));

    if packcurrent(i) < 0
      regen_energy = regen_energy - packvoltage(i) * packcurrent(i) * (t(i) - t(i-1));
    end

    %{
    packpower(i)
    packvoltage(i-1)
    packvoltage(i)
    packcurrent(i)
    cellv
    Ah_used
    %}
end

fprintf("== one lap ==\n");
fprintf("duration:\t%.2f s\n", t(end));
fprintf("Energy consumed:\t%.3f kWh\n", energy_used / (3600 * 1000));
fprintf("Energy regen:\t%.3f kWh\n", regen_energy / (3600 * 1000));
fprintf("Pack power avg.:\t%.3f kW\n", mean(packpower) / 1000);
fprintf("Pack power abs. avg.:\t%.3f kW\n", mean(abs(packpower)) / 1000);
fprintf("Current avg.:\t%.3f A\n", mean(packcurrent));
fprintf("Current abs. avg.:\t%.3f A\n", mean(abs(packcurrent)));
fprintf("Current RMS:\t%.3f A\n", rms(packcurrent));
fprintf("== over 35 laps ==\n");
fprintf("Total energy consumed:\t%.3f kWh\n", energy_used / (3600 * 1000) * 35);

packheat = (packcurrent).^ 2 * packIR;
cellheat = (packcurrent).^ 2 * packconfig(1) / packconfig(2) * cellIR;

% inverter
% note: mathematics done after the pack because the efficiency correlation
% was done on input voltage and current. These are determined from pack
% numbers, so they must be done first.
inverterheat = inverter_loss(packvoltage, abs(packcurrent));
inverterheatmean = mean(inverterheat, "omitnan");

% == endurance score == 
% using 2026 results
t_min = 1312;
t_max = t_min * 1.45; 
e_max = 0.797;
e_min = 0.289;
E_min = 2.84; %kWh

% score
t_us = t(end) * 32;
score_time = 250 * (max(t_max, t_us) / t_us - 1) / 0.45;
score_laps = 25; % assume we finish XD
E_us = energy_used / (3600 * 1000) * 35;
e_us = (min(t_us, t_min) / t_us) * (min(E_us, E_min) / E_us);
score_eff = 100 * (e_us  - min(e_min, e_us) ) / (max(e_us, e_max) - min(e_min, e_us));

fprintf("== endurance score (based on 2026) ==\n");
fprintf("time:\t%.2f s\n", t_us);
fprintf("energy:\t%.2f kWh\n", E_us);
fprintf("e factor:\t%.2f\n", e_us);
fprintf("time score:\t%.2f\n", score_time);
fprintf("lap  score:\t25\n");
fprintf("eff  score:\t%.2f\n", score_eff);
fprintf("total score:\t%.2f\n", score_time + score_eff + 25);

% == figure 1: Driver input ==
figure;
subplot(2,1,1);
hold on;
legend('-DynamicLegend');
plot(t, wheelpower, 'g--', DisplayName="wheel");
plot(t, packpower, 'g-', DisplayName='pack');
plot(t, -packpower, 'c-', DisplayName='regen', LineWidth=1);
yline(80000, 'k--', DisplayName="rules limit");
yline(packpowermean, 'k-', DisplayName="pack mean");
ylim([0, 100000]);
xlabel("time(s)");
ylabel("power(W)");

% subplot(3,1,2);
% plot(t, torque, 'b-');
% ylim([-50, 150]);
% xlabel("time(s)");
% ylabel("torque(Nm)");

subplot(2,1,2);
hold on;
yyaxis left;
plot(t, rpm, 'r-');
yline(mean(rpm));
xlabel("time(s)");
ylabel("rpm");

yyaxis right;
plot(t, accel_long, 'r--');
ylabel("acceleration (G)");
legend('rpm', 'mean rpm', 'long. accel.');
sgtitle(filename)

% == figure 2: Accu stats ===
figure;
subplot(2,1,1);
legend('-DynamicLegend');
hold on;
yyaxis left;
plot(t, packcurrent, 'b-', DisplayName="current");
yline(mean(abs(packcurrent)), 'b-', DisplayName=sprintf("abs. mean: %.0f", mean(abs(packcurrent))));
yline(0, 'k--', HandleVisibility="off");
yline(rms(packcurrent), 'k-', DisplayName=sprintf("rms: %.0f", rms(packcurrent)));
xlabel("time(s)");
ylabel("current(A)");
yyaxis right;
plot(t, packvoltage, 'g-', DisplayName="voltage");
ylabel("voltage(V)");
ylim([200, 500]);

subplot(2,1,2);
legend('-DynamicLegend');
hold on;

plot(t, packheat, 'r--', DisplayName='pack');
plot(t, cellheat, 'r-', DisplayName='cells');
yline(mean(cellheat), 'k-', DisplayName=sprintf("cells mean: %.0f", mean(cellheat)));

xlabel("time(s)");
ylabel("pack heat(W)");
sgtitle(filename)

% == figure 3: Tractive load ==
figure;
subplot(2,1,1);
legend('-DynamicLegend');
hold on;
yyaxis left;
plot(t, motorheat, 'r-', HandleVisibility='off');
yline(motorheatmean, 'k-', DisplayName=sprintf("mean: %.0f", motorheatmean));
ylim([0, 3000]);
ylabel("motor heat(W)");
xlabel("time(s)");

yyaxis right;
plot(t, motoreff, 'g-', DisplayName="efficiency");
ylim([0, 1.2]);
ylabel("efficiency(%)");

subplot(2,1,2);
legend('-DynamicLegend');
hold on;
plot(t, inverterheat, 'r-', HandleVisibility='off');
yline(inverterheatmean, 'k-', DisplayName=sprintf("mean: %.0f", inverterheatmean));
ylim([0, 1500]);
ylabel("inverter heat(W)");
xlabel("time(s)");
sgtitle(filename)
