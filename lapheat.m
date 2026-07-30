% ===
% This script uses Assetto Corsa Simulation Driver input & engine data to 
% determine powertrain thermal losses over one lap
% ===

clear; clc; close all;

% first three rows are empty
% raw = readmatrix("datasets/lapheat_autox_loop_1_lap_90Nm.csv");
raw = readmatrix("datasets/lapheat_autox_loop_1_lap_116Nm.csv");

% extract
t = raw(:, 1);
rpm = raw(:,2);
torque = raw(:,3);
throttle = raw(:,4) / 100;

% grab interpolations
motor_efficiency=motor_efficiency();
sony_voltage=sony_voltage();

% properties
drivetrainefficiency = 0.928;
moduleIR = 0.002; % from Nandana's 2023 calorimetry. ENEPAQ 6p module
packIR = moduleIR * 100 * 1.2; % assume +20% from bus bars and others
packconfig = [100, 6];

% these refer to the power going OUT of the components they are named.
wheelpower = torque .* rpm * 2 * pi / 60;
motorpower = wheelpower / drivetrainefficiency;

% motor
motoreff = arrayfun(@(a, b) motor_efficiency(a,b), rpm, torque);
motorheat = max(0, motorpower) .* (1-motoreff);
motorheatmean = mean(motorheat);
inverterpower = motorpower ./ motoreff;

% pack
% pack power cannot account for inverter losses because inverter losses are
% correlated to input voltage and current. chicken and egg problem. see
% note in inverter section.
packpower = inverterpower;
packvoltage = zeros(1, length(t));
packcurrent = zeros(1, length(t));
packvoltage(1) = 420;

Ah_used = 0;
energy_used = 0; % Joules
for i = 2:length(t)
    packcurrent(i) = packpower(i) / packvoltage(i-1);
    Ah_used = Ah_used + packcurrent(i) * (t(i) - t(i-1)) / 3600;
    cellv = sony_voltage(Ah_used / packconfig(2));

    packvoltage(i) = cellv * packconfig(1);
    energy_used = energy_used + packvoltage(i) * packcurrent(i) * (t(i) - t(i-1));

    %{
    packpower(i)
    packvoltage(i-1)
    packvoltage(i)
    packcurrent(i)
    cellv
    Ah_used
    %}
end
fprintf("Ah used: %.3f Ah\n", Ah_used);
fprintf("Total energy consumed: %.3f kWh\n", energy_used / (3600 * 1000));
fprintf("== over 32 laps ==\n");
fprintf("Ah used: %.3f Ah\n", Ah_used * 32);
fprintf("Total energy consumed: %.3f kWh\n", energy_used / (3600 * 1000) * 32);

packcurrentmean = mean(abs(packcurrent));
packheat = (packcurrent).^ 2 * packIR;
packheatmean = mean(packheat);

% inverter
% note: mathematics done after the pack because the efficiency correlation
% was done on input voltage and current. These are determined from pack
% numbers, so they must be done first.
inverterheat = inverter_loss(packvoltage, abs(packcurrent));
inverterheatmean = mean(inverterheat);

% heat metric for estimations.
% average power squared is helpful for heat maths. see prints below.
avg_p2 = sum(packpower .^ 2 * (t(2) - t(1))) / t(end);

fprintf("== heat metric ==\n");
fprintf("q = I^2 R\n");
fprintf("q = (P/V)^2 R\n");
fprintf("q = R/V^2 * P^2 \n");
fprintf("q_avg = R/V^2 * (P^2)_avg \n");
fprintf("1/t * int P^2 dt: %e J^2\n", avg_p2);



% == figure 1: Driver input ==
figure;
subplot(3,1,1);
hold on;
plot(t, wheelpower, 'g--');
plot(t, packpower, 'g-');
plot(t, packvoltage*90, 'k--');
legend("wheel", "pack", "fuse");
xlabel("time(s)");
ylabel("power(W)");

subplot(3,1,2);
plot(t, torque, 'b-');
ylim([-50, 150]);
xlabel("time(s)");
ylabel("torque(Nm)");

subplot(3,1,3);
hold on;
yyaxis left;
plot(t, rpm, 'r-');
plot(t, 5500*(torque > 100), 'b--');
yline(mean(rpm));
xlabel("time(s)");
ylabel("rpm");

yyaxis right;
plot(t, rpm * pi * 16 * 60 / 63360 / 4.91, 'r--');
ylabel("mph (fdr=4.91)");
legend('rpm', 'max torque request', 'mean rpm', 'speed');

% == figure 2: Accu stats ===
figure;
subplot(2,1,1);
legend('-DynamicLegend');
hold on;
yyaxis left;
plot(t, packcurrent, 'b-', DisplayName="current");
yline(packcurrentmean, 'b-', DisplayName=sprintf("mean: %.0f", packcurrentmean));
yline(0, 'k--', HandleVisibility="off");
yline(90, 'k-', DisplayName="fuse");
xlabel("time(s)");
ylabel("current(A)");
yyaxis right;
plot(t, packvoltage, 'g-', DisplayName="voltage");
ylabel("voltage(V)");
ylim([220, 420]);

subplot(2,1,2);
legend('-DynamicLegend');
hold on;

plot(t, packheat, 'r-', HandleVisibility='off');
yline(packheatmean, 'k-', DisplayName=sprintf("mean: %.0f", packheatmean));

xlabel("time(s)");
ylabel("pack heat(W)");

% == figure 3: Tractive load ==
figure;
subplot(2,1,1);
legend('-DynamicLegend');
hold on;
yyaxis left;
plot(t, motorheat, 'r-', HandleVisibility='off');
yline(motorheatmean, 'k-', DisplayName=sprintf("mean: %.0f", motorheatmean));
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
ylabel("inverter heat(W)");
xlabel("time(s)");
