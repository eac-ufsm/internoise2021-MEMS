%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sound pressure level calculations - Code example
%
% Part of the article ''MEMS digital microphone and Arduino compatible 
%  microcontroller: an embedded system for noise monitoring'' 
%  presented at Internoise 2021 (Washington DC).
%
% You should run cell by cell (ctrl + shift + enter) to visualize the RMS 
%  calculations step by step.
%
% Felipe Ramos de Mello, William D'Andrea Fonseca and Paulo Henrique Mareze
%
% 31/05/2021
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Cleaning service
clear all; close all; clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Generating a sine wave

% Basic parameters
frequency = 1000;     % Frequency in Hertz
period = 1/frequency; % Period in seconds
w = 2*pi*frequency;   % Angular frequency (rad/s)
A = 1;                % Amplitude in Pascals (RMS value should be ~0.707)

% Audio parameters
fs = 44100;             % Sampling rate
duration = 10;          % In seconds
nSamples = fs*duration; % Number of samples

% Sound wave
timeVector = 0:1/fs:duration - 1/fs;
sineWave = A*sin(w*timeVector);

% Plot (4 periods)
figure('Name', 'sinewave', 'DefaultAxesFontSize', 18, 'OuterPosition', [100 100 1200 750])
plot(timeVector, sineWave, 'LineWidth', 1.5); hold on;

title('1 kHz sine wave')
xlabel('Time [s]');
ylabel('Amplitude [-]');
legend('Sine wave', 'location', 'south');

xlim([0, period*4]);
grid on;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% RMS calculation - Pt 1

% First we square all samples
sineWaveSquared = sineWave.^2;

plot(timeVector, sineWaveSquared, 'LineWidth', 2); hold on;
legend('Sine wave', 'Squared sine wave', 'location', 'south');

%% RMS calculation - Pt 2

% Next we calculate the mean of the square values
meanSquare = sum(sineWaveSquared)/nSamples;

yline(meanSquare, '--', 'LineWidth', 2);
legend('Sine wave', 'Squared sine wave', sprintf('Mean of the squared sine wave: %.3f', meanSquare),...
    'location', 'south');

%% RMS calculation - Pt 3

% Finally, we apply the square root
rootMeanSquare = sqrt(meanSquare);

yline(rootMeanSquare, 'Color', 'g', 'LineWidth', 4); hold on;
legend('Sine wave', 'Squared sine wave', sprintf('Mean of the squared sine wave: %.3f', meanSquare),...
    sprintf('Root mean square value: %.3f', rootMeanSquare), 'location', 'south');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SPL of the sine wave

% With the RMS value, one can calculate the SPL by the equation
% 20*log10(Prms/Pref), with Pref = 2e-5 Pa.

p_ref = 2e-5; % Reference pressure in Pascals
SPL = 20*log10(rootMeanSquare/p_ref); % Sound Pressure Level
plot(NaN, NaN, 'color', 'w');
legend('Sine wave', 'Squared sine wave', sprintf('Mean of the squared sine wave: %.3f', meanSquare),...
    sprintf('Root mean square value: %.3f', rootMeanSquare),...
    sprintf('SPL: %.3f dB', SPL), 'location', 'south');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Clear everything for the next task

clear all; close all; clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SPL calculations for a complex sound

% Let's create a special signal and learn how to calculate SPL values such
% as LZeq, LAeq, LCeq, Lpeak, and so on. Our signal is a broadband sound
% comprised of 6 parts, each one with a diferent level.

% Audio parameters
signalGeneration.duration = 5; % Duration of each part in seconds
fs = 44100;   % Sampling rate
signalGeneration.nSamples = fs*signalGeneration.duration;
timeVector = 0:1/fs:30-1/fs;

% Energies for each part of the signal
signalGeneration.A1 = 0.10; % RMS value for the first part
signalGeneration.A2 = 0.60; % RMS value for the second part
signalGeneration.A3 = 1.00; % RMS value for the third part
signalGeneration.A4 = 0.70; % RMS value for the fourth part
signalGeneration.A5 = 0.30; % RMS value for the fifth part
signalGeneration.A6 = 0.01; % RMS value for the sixth part

% Creating 6 broadband signals with diferent amplitudes
signalGeneration.pt1 = signalGeneration.A1*randn(signalGeneration.nSamples, 1);
signalGeneration.pt2 = signalGeneration.A2*randn(signalGeneration.nSamples, 1);
signalGeneration.pt3 = signalGeneration.A3*randn(signalGeneration.nSamples, 1);
signalGeneration.pt4 = signalGeneration.A4*randn(signalGeneration.nSamples, 1);
signalGeneration.pt5 = signalGeneration.A5*randn(signalGeneration.nSamples, 1);
signalGeneration.pt6 = signalGeneration.A6*randn(signalGeneration.nSamples, 1);

signal = [signalGeneration.pt1; signalGeneration.pt2; signalGeneration.pt3;...
    signalGeneration.pt4; signalGeneration.pt5; signalGeneration.pt6];

% Visualization
figure('Name', 'Complex Signal', 'DefaultAxesFontSize', 18, 'OuterPosition', [100 100 1200 750]);
plot(timeVector, signal);

xlabel('Time [s]');
ylabel('Amplitude [Pa]');
title('Broadband random noise');

grid on;

% Uncomment to listen - check out for your system's volume!
% sound(signal/max(abs(signal)), fs); % The signal was normalized
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Equivalent continuous sound pressure level (Leq), peak values and time weighted SPL

% First, lets create two filter objects in order to apply the frequency
% weightings A and C
A_weighting = weightingFilter('A-weighting', fs);
C_weighting = weightingFilter('C-weighting', fs);

% Filters visualization (uncomment to see their frequency responses)
% A_weighting.visualize;
% C_weighting.visualize;

% Time weighting filters
fastWeighting = timeWeighting('Fast', fs);
slowWeighting = timeWeighting('Slow', fs);

% Now apply the frequency weightings to the signal
signal_A = A_weighting(signal);
signal_C = C_weighting(signal);

% For time weighted SPL we must apply the time weighting filters to the
% frequency weighted squared signal
signal_A_fast = fastWeighting(signal_A.^2);
signal_C_fast = fastWeighting(signal_C.^2);
signal_Z_fast = fastWeighting(signal.^2);

signal_A_slow = slowWeighting(signal_A.^2);
signal_C_slow = slowWeighting(signal_C.^2);
signal_Z_slow = slowWeighting(signal.^2);

% Next we calculate the RMS values for each signal (for Leq values)
Prms_A = rms(signal_A);
Prms_C = rms(signal_C);
Prms_Z = rms(signal);

% Finally, we calculate the Leq and time-weighted SPL
p_ref = 2e-5;
LAeq = 20*log10(Prms_A/p_ref);
LCeq = 20*log10(Prms_C/p_ref);
LZeq = 20*log10(Prms_Z/p_ref);

% Fast
LAF = 10*log10(signal_A_fast/p_ref.^2); % Remember that the signal_A_fast is already squared
LCF = 10*log10(signal_C_fast/p_ref.^2);
LZF = 10*log10(signal_Z_fast/p_ref.^2);
% Slow
LAS = 10*log10(signal_A_slow/p_ref.^2); % Remember that the signal_A_fast is already squared
LCS = 10*log10(signal_C_slow/p_ref.^2);
LZS = 10*log10(signal_Z_slow/p_ref.^2);

% Maximum time-weighted SPL values
LAFmax = max(LAF); LCFmax = max(LCF); LZFmax = max(LZF);
LASmax = max(LAS); LCSmax = max(LCS); LZSmax = max(LZS);

% Time-weighted spl plot 
figure('Name', 'Complex Signal', 'DefaultAxesFontSize', 18, 'OuterPosition', [100 100 1200 750]);
plot(timeVector, LAF, 'LineWidth', 2); hold on;
plot(timeVector, LAS, 'LineWidth', 2);

xlabel('Time [s]');
ylabel(['SPL [dB ref. 20 ' char(181) 'Pa]']);
title('Time-weighted SPL for the broadband random noise');
legend('Fast-weighted', 'Slow-weighted', 'location', 'south', 'NumColumns', 2);

grid on;

clc;
fprintf('Equivalent continuous sound pressure levels:\n');
% Print Leq values
fprintf('\nLAeq = %.2f dB\nLCeq = %.2f dB\nLZeq = %.2f dB\n', LAeq, LCeq, LZeq);

fprintf('\nTime-weighted maximum SPL:\n');
% Print LFmax and LSmax values
fprintf(['\nLAFmax = %.2f dB\tLASmax = %.2f dB\n',....
    'LCFmax = %.2f dB\tLCSmax = %.2f dB\n',...
    'LZFmax = %.2f dB\tLZSmax = %.2f dB\n'], LAFmax, LASmax,...
    LCFmax, LCSmax, LZFmax, LZSmax);

fprintf('\nPeak SPL:\n');
% For peak values we seek for the maximum values of each signal
Peak_A = max(abs(signal_A));
Peak_C = max(abs(signal_C));
Peak_Z = max(abs(signal));

% Lpeak values
LApeak = 20*log10(Peak_A/p_ref);
LCpeak = 20*log10(Peak_C/p_ref);
LZpeak = 20*log10(Peak_Z/p_ref);

% Print peak values
fprintf('\nLApeak = %.2f dB\nLCpeak = %.2f dB\nLZpeak = %.2f dB\n', LApeak, LCpeak, LZpeak);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Equivalent continous sound pressure level integrated over a period T

% Leq can be calculated for different integration period. Here we will see
% how to do this and how to retrieve the Leq for the whole measurement.

T = 5; % Integration period in seconds (you can change this value and see how it affects the calculations)
samplesPerIntegration = T*fs; % Number of samples in T seconds

% Counters
sampleCounter1 = 1; % First sample index
sampleCounter2 = samplesPerIntegration; % Last sample index
i = 1; % numLogs counter
numLogs = floor(length(signal)/samplesPerIntegration); % Total averages

% Pre allocation of the Leq vectors
LAeq_T = zeros(numLogs, 1);
LCeq_T = zeros(numLogs, 1);
LZeq_T = zeros(numLogs, 1);

% Block calculations
while sampleCounter1 < length(signal)
    
    % Chunk of the signals (T seconds long)
    chunk_A = signal_A(sampleCounter1:sampleCounter2);
    chunk_C = signal_C(sampleCounter1:sampleCounter2);
    chunk_Z = signal(sampleCounter1:sampleCounter2);
    
    % Leq for each chunk
    LAeq_T(i) = 20*log10(rms(chunk_A)/p_ref);
    LCeq_T(i) = 20*log10(rms(chunk_C)/p_ref);
    LZeq_T(i) = 20*log10(rms(chunk_Z)/p_ref);
    
    % Update the counters
    sampleCounter1 = sampleCounter2 + 1;
    sampleCounter2 = sampleCounter2 + samplesPerIntegration;
    i = i + 1;
    
end

% Calculation of the global values (for the whole measurement)
LAeq_global = 10*log10(mean(10.^(LAeq_T/10)));
LCeq_global = 10*log10(mean(10.^(LCeq_T/10)));
LZeq_global = 10*log10(mean(10.^(LZeq_T/10)));

% Visualization
figure('Name', 'Leq_T', 'DefaultAxesFontSize', 18, 'OuterPosition', [100 100 1200 750]);

stairs(timeVector(1:samplesPerIntegration:end), LAeq_T, 'LineWidth', 2); hold on;
stairs(timeVector(1:samplesPerIntegration:end), LCeq_T, 'LineWidth', 2); hold on;
stairs(timeVector(1:samplesPerIntegration:end), LZeq_T, 'LineWidth', 2); hold on;
plot(NaN, NaN, 'color', 'w'); hold on;
plot(NaN, NaN, 'color', 'w'); hold on;
plot(NaN, NaN, 'color', 'w'); hold on;

title(sprintf('Leq integrated every %.1f seconds', T));
xlabel('Time [s]');
ylabel(['SPL [dB ref. 20' char(181) 'Pa]']);

legend('LAeq,T', 'LCeq,T', 'LZeq,T' ,...
    sprintf('LAeq: %.2f dB', LAeq_global) , sprintf('LCeq: %.2f dB', LCeq_global),...
    sprintf('LZeq: %.2f dB', LZeq_global),...
    'NumColumns', 2, 'Location', 'South');

grid on;

% Comparison between the block calculations and calculations for the whole
% signal
clc;
fprintf('Equivalent continuous SPL calculated for the whole signal:\n');
% Print Leq values
fprintf('\nLAeq = %.2f dB\nLCeq = %.2f dB\nLZeq = %.2f dB\n', LAeq, LCeq, LZeq);

fprintf('\nEquivalent continuous SPL calculated in blocks:\n');
% Print Leq values
fprintf('\nLAeq = %.2f dB\nLCeq = %.2f dB\nLZeq = %.2f dB\n', LAeq_global, LCeq_global, LZeq_global);

fprintf('\n It seems that everything went alright! The values must be the same.\n');
fprintf('\nYou should try change the integration time and see how it goes...\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Changing the integration period

% We calculated the Leq values for a integration period of 5s. However, if
% you want to change the period, it is possible by the following method.
% Let's try for a 10s period.

new_T = 10;
ratio = new_T/T;
new_numLogs = numLogs/ratio;
new_Leq = zeros(new_numLogs, 1);
k = 1;

for j = 1:new_numLogs
    new_Leq(j) = 10*log10(mean(10.^(LZeq_T(k:k+1)/10)));
    k = k + 2;
end

% Now let's compare with the Leq for T = 5s

figure('Name', 'Leq_T', 'DefaultAxesFontSize', 18, 'OuterPosition', [100 100 1200 750]);

stairs([timeVector(1:samplesPerIntegration:end) max(timeVector)], [LZeq_T; LZeq_T(end)], 'LineWidth', 2); hold on;
stairs([timeVector(1:samplesPerIntegration*2:end) max(timeVector)], [new_Leq; new_Leq(end)], 'LineWidth', 2); hold on;

title(sprintf('Leq integrated every %.1f and %.1f seconds', T, new_T));
xlabel('Time [s]');
ylabel(['SPL [dB ref. 20' char(181) 'Pa]']);

legend('LZeq,5', 'LZeq,10',...
    'NumColumns', 2, 'Location', 'South');

grid on;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Leq in octave bands

% first we generate our filter bank
n = 1; % Fractional octave bands number (1/n) - you should try 1 or 3
freqRange = [15, 20000];
[filterBank, Fc] = octaveFilterBank(n, fs, freqRange); % 1/n, sample rate and frequency range

% Pre-allocation for the filtered signals
octA = zeros(length(signal), length(Fc));
octC = zeros(length(signal), length(Fc));
octZ = zeros(length(signal), length(Fc));

% Next we apply the filters to our frequency-weighted signals
for j = 1:length(Fc)
    octA(:, j) = filterBank{j}(signal_A);
    release(filterBank{j});
    octC(:, j) = filterBank{j}(signal_C);
    release(filterBank{j});
    octZ(:, j) = filterBank{j}(signal);
    release(filterBank{j})
end

% Now we calculate the SPL for each band
LAoct = 20*log10(rms(octA)/p_ref);
LCoct = 20*log10(rms(octC)/p_ref);
LZoct = 20*log10(rms(octZ)/p_ref);

% String for the frequency ticks
switch n
    case 1
        str = {'16', '32', '63', '125', '250', '500', '1000', '2000', '4000', '8000', '16000'};
        x = 1:length(Fc);
    case 3
        str = {'16', '32', '63', '125', '250', '500', '1000', '2000', '4000', '8000', '16000', '20000'};
        x = 1:3:length(Fc) + 3;
end

% Visualization
figure('Name', 'Octave', 'DefaultAxesFontSize', 18, 'OuterPosition', [100 100 1200 750]);

bar([LAoct' LCoct' LZoct']);
xticks(x); xticklabels(str);

title(sprintf('Equivalent continuous SPL in 1/%.0f octave bands', n))
xlabel('Frequency bands [Hz]');
ylabel(['SPL [dB ref. 20 ' char(181) 'Pa]'])
legend('A-weighted', 'C-weighted', 'Z-weighted', 'location', 'south', 'NumColumns', 3)

grid on;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Comparison between Leq calculated for the whole signal, in blocks, and 
%   by the sum of octave bands

LAeq_oct_global = 10*log10(sum(10.^(LAoct/10)));
LCeq_oct_global = 10*log10(sum(10.^(LCoct/10)));
LZeq_oct_global = 10*log10(sum(10.^(LZoct/10)));

clc;
fprintf('Equivalent continuous SPL calculated for the whole signal:\n');
% Print Leq values
fprintf('\nLAeq = %.2f dB\nLCeq = %.2f dB\nLZeq = %.2f dB\n', LAeq, LCeq, LZeq);

fprintf('\nEquivalent continuous SPL calculated in blocks:\n');
% Print Leq values
fprintf('\nLAeq = %.2f dB\nLCeq = %.2f dB\nLZeq = %.2f dB\n', LAeq_global, LCeq_global, LZeq_global);

fprintf('\nEquivalent continuous SPL calculated by the sum of octave bands:\n');
% Print Leq values
fprintf('\nLAeq = %.2f dB\nLCeq = %.2f dB\nLZeq = %.2f dB\n', LAeq_oct_global, LCeq_oct_global, LZeq_oct_global);

fprintf('\nIt seems that everything went alright! The values shloud be around the same.\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Cleaning service 

clear all; clc; close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Spectrum and fractional octave band analysis

% It should be noted that, in this case, the octave bands SPL are equal to
% the Leq for the whole signal. Furthermore, the octave band levels increses
% across the frequency due to the fact that they are the level sum for all
% frequencies inside the band. Ultimately, the sum of each band and the sum
% of the whole spectrum should return around the same value.

fs = 44100;   % Sample rate
p_ref = 2e-5; % Reference pressure

% Octave filter banks
[filterBank1, fc1] = octaveFilterBank(1, fs, [10, 20000]);
[filterBank3, fc3] = octaveFilterBank(3, fs, [10, 20000]);

% Generates a simple white noise (5 seconds long)
signal = wgn(fs*5, 1, 1);

% Octave filtering
Loct1 = zeros(length(fc1), 1);
Loct3 = zeros(length(fc3), 1);

for j = 1:length(fc1)
    dummy = filterBank1{j}(signal);
    Loct1(j) = 20*log10(rms(dummy)/p_ref);
end
   
for j = 1:length(fc3)
    dummy = filterBank3{j}(signal);
    Loct3(j) = 20*log10(rms(dummy)/p_ref);
end
 
% Single sided power spectrum estimation using fft
NFFT = length(signal); % Number of fft points
Y = fft(signal, NFFT); % FFT calculation
Y = sqrt(2).*Y/NFFT; % Amplitude adjustment (rms value)
Y = Y.*conj(Y); % Magnitude calculation
Y = Y(1:NFFT/2+1); % Select just the positive frequencies

freqVector = (fs/NFFT)*(0:NFFT/2); % Frequency vector -> (NFFT/fs) is the resolution in frequency domain
LP = 10*log10(Y/p_ref.^2);    % Calculates SPL for the spectrum 

% Calculates the edge frequencies for the octave band filters
G = 10^(3/10);
fc_stairs1 = fc1*(G^(-1/(2*1)));
fc_stairs1 = [fc_stairs1 fc1(end)*(G^(1/(2*1)))];
fc_stairs3 = fc3*(G^(-1/(2*3)));
fc_stairs3 = [fc_stairs3 fc3(end)*(G^(1/(2*3)))];

figure('Name', 'Spectrum', 'DefaultAxesFontSize', 18, 'OuterPosition', [0 100 1500 750]);

semilogx(freqVector, LP, 'LineWidth', 1.5); hold on;
stairs(fc_stairs1, [Loct1; Loct1(end)], 'LineWidth', 1.5, 'marker', 'd');
stairs(fc_stairs3, [Loct3; Loct3(end)], 'LineWidth', 1.5, 'marker', 'o');

xlim([11.22 20e3])

title('Spectrum SPL, 1/1 octave bands SPL and 1/3 octave bands SPL');
ylabel(['SPL [dB ref. 20 ' char(181) 'Pa]'])
xlabel('Frequency [Hz]');

legend('Spectrum', '1/1 octave bands', '1/3 octave bands', 'NumColumns', 3, 'location', 'south');

grid on;

Lspectrum_global = 10*log10(sum(10.^(LP/10)));
Loct1_global = 10*log10(sum(10.^(Loct1/10)));
Loct3_global = 10*log10(sum(10.^(Loct3/10)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SPL vs Leq vs Time domain

% With sound level meter it is possible to assess both SPL and Leq values.
% The difference is that the first is usually time-weighted and used to
% display values at the screen, as well as for Lmax calcultions 
% (some normatives use this parameter to caraterize impulsive sounds). 
% The second is the most used parameter and
% evaluates the sound pressure level over a integration period, T. 

% A Sound Level Meter usually display time-weighted SPL at a 1s update rate
% by default. However, internally it is calculating the SPL for each sample
% that is captured by the microphone. In parallel, it evaluates Leq
% values integrated over a period defined by the user (common values are
% 1s, 10s, 30s, 60s, 10min, 1h) and stores them each time a new value is
% available. Here we try to emulate this for smaller periods. We are
% considering:
%
%   1 - Instant time-weighted SPL calculated for each sample;
%   2 - LZfast (time-weighted SPL) updated every 0.125s;
%   3 - LZeq integrated over 0.5s.

filter = timeWeighting('Fast', fs);

signal = [wgn(fs*0.2, 1, 0.001); wgn(fs*0.3, 1, 1); wgn(fs*0.2, 1, 0.003); wgn(fs*0.4, 1, 0.00001);...
          wgn(fs*0.4, 1, 0.7); wgn(fs*0.5, 1, 3); wgn(fs*0.5, 1, 0.00001);...
          wgn(fs*0.2, 1, 0.001); wgn(fs*0.3, 1, 1); wgn(fs*0.2, 1, 0.00003); wgn(fs*0.4, 1, 0.02);...
          wgn(fs*0.4, 1, 0.3); wgn(fs*0.5, 1, 0.015); wgn(fs*0.5, 1, 1)]; 

timeVector = 0:1/fs:length(signal)/fs - 1/fs;
SPL = 10*log10(filter(signal.^2)/p_ref.^2);

% Leq over a integration period T
T = 0.5; % Integration period in seconds (you can change this value and see how it affects the calculations)
samplesPerIntegration = T*fs; % Number of samples in T seconds

% Counters
sampleCounter1 = 1; % First sample index
sampleCounter2 = samplesPerIntegration; % Last sample index
i = 1; % numLogs counter
numLogs = floor(length(signal)/samplesPerIntegration); % Total averages

% Pre allocation of the Leq vector
Leq_T = zeros(numLogs, 1);
timeLeq = zeros(numLogs, 1);

while sampleCounter1 < length(signal)
    
    % Chunk of the signals (T seconds long)
    chunk_Z = signal(sampleCounter1:sampleCounter2);
    
    % Leq for each chunk
    Leq_T(i) = 20*log10(rms(chunk_Z)/p_ref);
    
    timeLeq(i) = timeVector(sampleCounter1) + T;
    
    % Update the counters
    sampleCounter1 = sampleCounter2 + 1;
    sampleCounter2 = sampleCounter2 + samplesPerIntegration;
    i = i + 1;
    
end

figure('Name', 'Figure', 'DefaultAxesFontSize', 18, 'OuterPosition', [100 100 1200 750]);

plot(timeVector, SPL, 'LineWidth', 2); hold on;
stairs(timeLeq, Leq_T, 'LineWidth', 2); hold on;
plot(timeVector(1:floor(0.125*fs):end), SPL(1:floor(0.125*fs):end), '*', 'LineWidth', 8);

title('Instant SPL, LZ_{fast} and LZ_{eq} for a random signal')
xlabel('Time [s]');
ylabel(['SPL [dB ref. 20 ' char(181) 'Pa]'])
legend('Instant fast-weighted SPL', 'LZ_{fast} evaluated every 0.125 seconds', 'LZ_{eq} integrated over 0.5 seconds');

ylim([93, 98])
grid on;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Functions
function [filterBank, Fc] = octaveFilterBank(n, fs, freqRange)

% Defines fractional octave bands
switch n
    case 1
        bandwidth = '1 Octave';
        N = 10; %10
    case 3
        bandwidth = '1/3 Octave';
        N = 10; %10
end

Fc = centerFreqsCalculator(n, freqRange);
Fc(Fc > freqRange(2)) = [];
Fc(Fc < freqRange(1)) = [];
Nfc = length(Fc); % Number of pass-band filters to be generated
filterBank = cell(1,Nfc); % Cell to store the filters

% Generates de octave filter bank based on the freqRange and center
% frequencies
for i = 1:Nfc   
    if Fc(i) < 16e3 
        filterBank{i} = octaveFilter('FilterOrder', N,...
            'CenterFrequency', Fc(i), 'Bandwidth', bandwidth, ...
            'SampleRate', fs);
    else
        filterBank{i} = octaveFilter('FilterOrder', N,...
            'CenterFrequency', Fc(i), 'Bandwidth', bandwidth, ...
            'SampleRate', fs);
        filterBank{i}.Oversample = true; % Para evitar problemas em frequências próximas a fs/2
    end    
end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculates octave bands centre frequencies based on ANSI S1.11-2004
function Fc = centerFreqsCalculator(n, freqRange)

% Base used for calculations 
G = 10^(3/10); % base 10
% G = 2; % Base 2

k = 1; % Counter
Fmax = 0; % Verifies if frequency reached the maximum
Fc = zeros(1, 500); % Stores the centre frequencies

% Calculations described on ANSI S1.11-2004
if rem(n, 2)~=0
    while Fmax < freqRange(2)
        Fc(k) = 1000*(G^((k-30)/n));
        k = k+1;
        Fmax = max(Fc);
    end

else
    while Fmax < freqRange(2)
        Fc(k) = 1000*(G^((2*k-59)/(2*n)));
        k = k+1;
        Fmax = max(Fc);
    end    
end    

% For the lowest frequencies
Fant = Fc(1);
Fc2 = zeros(1, 500);
k = 1;

while Fant > freqRange(1)
    Fc2(k) = Fant/(G^(1/n));
    Fant = Fc2(k);
    k = k+1;
end

Fc = [sort(Fc2) Fc];

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Time weighting filters for SPL calculations
function filter = timeWeighting(type, fs)

if strcmp(type, 'Fast')
        
    timeConstant = 0.125; % Time constant TAU
    b = 1/(timeConstant*fs); % Filter coeficients
    a = [1 -exp(-1/(timeConstant*fs))]; % Filter coeficients
    [sos, g] = tf2sos(b, a); % SOS matrix for the biquad filter
    
    filter = dsp.BiquadFilter; % Generate a biquad filter
    filter.SOSMatrix = sos; % Sets the filter's SOS matrix
    filter.ScaleValues = [1 g]; % Adjusts the filter's gain
    
elseif strcmp(type, 'Slow')
       
    timeConstant = 1; % Slow
    b = 1/(timeConstant*fs); % Filter coeficients
    a = [1 -exp(-1/(timeConstant*fs))]; % Filter coeficients
    [sos, g] = tf2sos(b, a); % SOS matrix for the biquad filter
    
    filter = dsp.BiquadFilter; % Generate a biquad filter
    filter.SOSMatrix = sos; % Sets the filter's SOS matrix
    filter.ScaleValues = [1 g]; % Adjusts the filter's gain
    
end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EOF
