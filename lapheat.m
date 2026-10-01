% ===
% This script uses Assetto Corsa Simulation Driver input & engine data to 
% determine powertrain thermal losses over one lap
% ===

clear; clc; 

% first three rows are empty
% raw = readmatrix("datasets/lapheat_autox_loop_1_lap_116Nm.csv");
raw = readmatrix("datasets/lapheat_autox_MIS_1_lap_80kW.csv");

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
%moduleIR = 0.002; % from Nandana's 2023 calorimetry. ENEPAQ 6p module
moduleIR = 0.0028; % from 17mOhm over 6p
packIR = moduleIR * 100 * 1.2; % assume +20% from bus bars and others
packconfig = [114, 6];

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
packpowermean = mean(packpower);
packvoltage = zeros(1, length(t));
packcurrent = zeros(1, length(t));
packvoltage(1) = 420;

Ah_used = 0;
energy_used = 0; % Joules
for i = 2:length(t)
    cellv = sony_voltage(Ah_used / packconfig(2));
    packv = cellv * packconfig(1);
    
    if packpower(i) > 0
        r = roots([-packIR, packv, -packpower(i)]);
        packcurrent(i) = min(r);
    else 
        r = roots([packIR, packv, -packpower(i)]);
        packcurrent(i) = max(r);
    end

    if packcurrent(i) == 0
        packvoltage(i) = packv;
    else 
        packvoltage(i) = abs(packpower(i) / packcurrent(i));
    end

    
    Ah_used = Ah_used + packcurrent(i) * (t(i) - t(i-1)) / 3600;
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
fprintf("== one lap ==\n");
fprintf("Ah used: %.3f Ah\n", Ah_used);
fprintf("Average pack power: %.3f kW\n", mean(packpower) / 1000);
fprintf("Average absolute pack power: %.3f kW\n", mean(abs(packpower)) / 1000);
fprintf("Average current: %.3f A\n", mean(packcurrent));
fprintf("Average absolute current: %.3f A\n", mean(abs(packcurrent)));
fprintf("RMS current: %.3f A\n", rms(packcurrent));
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
inverterheatmean = mean(inverterheat, "omitnan");

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
yline(80000, 'k--');
yline(packpowermean, 'k-');
ylim([0, 100000]);
legend("wheel", "pack", "limit", "pack mean");
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
yline(mean(rpm));
xlabel("time(s)");
ylabel("rpm");

yyaxis right;
plot(t, rpm * pi * 16 * 60 / 63360 / 4.91, 'r--');
ylabel("mph (fdr=4.91)");
legend('rpm', 'mean rpm', 'speed');

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
ylim([200, 500]);

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
ylim([0, 10000]);
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
ylim([0, 10000]);
ylabel("inverter heat(W)");
xlabel("time(s)");
