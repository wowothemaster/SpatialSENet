clear; clc;
close all;
addpath(genpath('./'));
seed = 42; 
rng(seed);

%% ========================================================================
%  PART 0: Path Configuration 
% ========================================================================
% Python interpreter path
pyexe = 'D:\Anaconda_envs\envs\SPCUP\python.exe'; 
pyScript = 'infer_single.py';

% Audio file paths 
maleFilePath   = 'target_signal.wav';        
femaleFilePath = 'interference_signal.wav';

mixWavPath       = fullfile(pwd, 'mixture_signal.wav');   
processedWavPath = fullfile(pwd, 'processed_signal.wav'); 

% Final output MAT file path
out_mat = 'Task2_Reverberant_5dB.mat';

%% ========================================================================
%  PART 1: Generate Room Impulse Response
% ========================================================================
fprintf('>>> [1/3] Generating RIR...\n');

% 1-1) ULA Parameters
M           = 2;
delta       = 0.08;
c           = 340;
fs          = 16000;
xmic        = (0:-delta:-(M-1)*delta).';
xmic        = xmic - mean(xmic);
ymic        = zeros(length(xmic),1);
zmic        = zeros(length(xmic),1);
rmic_init   = [xmic,ymic,zmic];

% Room parameters
roomx      = 4.9; roomy = 4.9; roomz = 4.9;
lroom      = [roomx,roomy,roomz];
    
array_center   = [2.45, 2.45, 1.5];
rmic           = rmic_init + array_center; 
    
% Target source
theta_s    = 90; phi_s = 90;
rsrc       = [2.45, 3.45, 1.5]; 
    
% Interference source
theta_i    = 90; phi_i = 40;
rsrc_i     = [3.22, 3.06, 1.5];
T60_s      = 0.5;
    
impluseLen  = 4096;            
highpass = true;
win = 'hanning';
    
% Generate RIRs
Hs = hIRgen(fs, c, T60_s, impluseLen, highpass, win, lroom, rmic, rsrc);
Hi = hIRgen(fs, c, T60_s, impluseLen, highpass, win, lroom, rmic, rsrc_i);
HD = hIRgen(fs, c, 0, impluseLen, highpass, win, lroom, rmic, rsrc);
    
early_stop = 0.05*fs;
HE = zeros(impluseLen, M);
HR = zeros(impluseLen, M);
for m = 1:M
    HE(1:early_stop-1,m)=Hs(1:early_stop-1,m);
    HR(early_stop:end,m)=Hs(early_stop:end,m);
end


%% ========================================================================
%  PART 2: Generate Mixture Data
% ========================================================================
fprintf('>>> [2/3] Generating mixed audio data...\n');
durSec = 5;
durSmp = durSec * fs;
  
% Read audio
[x, fs0] = audioread(maleFilePath);
if size(x,2) > 1, x = x(:,1); end
[inter, fs0i] = audioread(femaleFilePath);
if size(inter,2) > 1, inter = inter(:,1); end

if fs0 ~= fs || fs0i ~= fs
    error('Sampling rate mismatch! Must be 16000Hz');
end

requiredMaleLen   = round(durSmp * fs0  / fs);
requiredFemaleLen = round(durSmp * fs0i / fs);
assert(length(x) >= requiredMaleLen, 'Insufficient target audio length');
assert(length(inter) >= requiredFemaleLen, 'Insufficient interference audio length');

% Resample and Crop
x = x(1:requiredMaleLen);
Sclean = resample(x, fs, fs0);
Sclean = Sclean(1:durSmp); 
inter = inter(1:requiredFemaleLen);
Sinter = resample(inter, fs, fs0i);
Sinter = Sinter(1:durSmp); 

% Generate multi-channel microphone signals
Signal_para.M        = M;
Signal_para.Lentime  = durSmp;
Signal_para.S_audio  = Sclean;
Signal_para.I_audio  = Sinter;
[Xall, Xclean, Xearly, Xrev, Xinter] = ...
    gener_mch_signal(Hs, HD, HE, HR, Hi, Signal_para);

% Add noise and mix
power_x = Xclean(:,1)' * Xclean(:,1);
pow_i   = Xinter(:,1)' * Xinter(:,1);
v_inter = sqrt(power_x / pow_i) * Xinter;   
v_white = randn(durSmp, M);
pow_n   = v_white(:,1)' * v_white(:,1);
v_white = sqrt(power_x / pow_n * 10^(-5/10)) * v_white; 
noise = v_inter + v_white;
Xall  = Xall + noise;

% Normalization
Xclean = normalizeAudio(Xclean);
Sclean_norm = normalizeAudio(Sclean); 
Sinter_norm = normalizeAudio(Sinter);
Xall   = normalizeAudio(Xall);

% Save temporary file for Python
audiowrite(mixWavPath, Xall, fs); 
fprintf('>>> Mixture wav saved. Preparing for Python inference...\n');

%% ========================================================================
%  PART 3: Process & Metrics & Final Save
% ========================================================================
fprintf('>>> [3/3] Calling Python for inference...\n');
cmd = sprintf('"%s" "%s" "%s" "%s"', pyexe, pyScript, mixWavPath, processedWavPath);

[status, output] = system(cmd);
if status ~= 0
    fprintf('Python Output Error:\n%s\n', output);
    error('Python script execution failed');
end

% Read processed signal
if ~exist(processedWavPath, 'file')
     error('Processed audio not found. Check Python output or paths.');
end
[enhanced, fs2] = audioread(processedWavPath);
if fs ~= fs2, error('Sampling rate mismatch!'); end

enhanced = enhanced(:, 1);
enhanced = enhanced(:);
clean_ref_for_metric = Xclean(:, 1); 

% Truncation and Alignment
L = min([length(clean_ref_for_metric), length(enhanced), length(Sclean_norm)]);
clean_ref_for_metric = clean_ref_for_metric(1:L);
enhanced             = enhanced(1:L);
Sclean_final         = Sclean_norm(1:L); 
Sinter_final         = Sinter_norm(1:L); 

% Read original mixture for alignment validation
[mix_read, ~]        = audioread(mixWavPath);
mix_final            = mix_read(1:L, :);

% Calculate Metrics
fprintf('>>> Calculating metrics...\n');

tempRefPath = fullfile(pwd, 'temp_ref_clean.wav');
audiowrite(tempRefPath, clean_ref_for_metric, fs);

% --- Other Metrics ---
pesq_score = pesq(char(tempRefPath), char(processedWavPath));
si_snr_score = si_snr(enhanced, clean_ref_for_metric);
osinr_score  = osinr(enhanced, clean_ref_for_metric);
stoi_score   = stoi(clean_ref_for_metric, enhanced, fs);
visqol_score = visqol(enhanced, clean_ref_for_metric, fs, Mode="speech");
if exist(tempRefPath, 'file'), delete(tempRefPath); end

fprintf('-----------------------------------------------------------------------\n');
fprintf('SI-SNR: %.2f | OSINR: %.2f | STOI: %.3f | PESQ: %.3f | ViSQOL: %.3f\n', ...
         si_snr_score, osinr_score, stoi_score, pesq_score, visqol_score);
fprintf('-----------------------------------------------------------------------\n');

%% ================== Build Final Structure ==================
% 1. Signals
target_signal       = Sclean_final;       
interference_signal = Sinter_final;
mixture_signal      = mix_final;
processed_signal    = enhanced;

% 2. RIR Data
rir_data = struct();
rir_data.Hs = Hs;
rir_data.Hi = Hi;
rir_data.HD = HD;
rir_data.c  = c;

% 3. Metrics
metrics = struct();
metrics.OSINR   = osinr_score;
metrics.STOI    = stoi_score;
metrics.SI_SNR  = si_snr_score; 
metrics.PESQ    = pesq_score;
metrics.VISQOL  = visqol_score;

% 4. Params
params = struct();
params.fs              = fs;
params.mic_positions   = rmic;   
params.array_spacing   = delta;  
params.source_azimuth  = [theta_s, phi_s];
params.source_position = rsrc;   
params.SNR_dB          = 5;      
params.SIR_dB          = 0;      

% Save MAT file
save(out_mat, ...
    'target_signal', ...
    'interference_signal', ...
    'mixture_signal', ...
    'rir_data', ...
    'processed_signal', ...
    'metrics', ...
    'params');
fprintf('>>> Results successfully saved to: %s\n', out_mat);

%% ================== LOCAL FUNCTIONS ==================
function audioData = normalizeAudio(audioData)
    maxAbsVal = max(abs(audioData(:)));
    if maxAbsVal > 1e-6
        audioData = audioData / maxAbsVal;
    end
end

function score = si_snr(enhanced, clean)
    enhanced = enhanced - mean(enhanced);
    clean = clean - mean(clean);
    alpha = (clean' * enhanced) / (clean' * clean + eps);
    s_target = alpha * clean;
    e_noise = enhanced - s_target;
    snr = (s_target' * s_target) / (e_noise' * e_noise + eps);
    score = 10 * log10(snr);
end

function score = osinr(enhanced, clean)
    enhanced = enhanced - mean(enhanced);
    clean = clean - mean(clean);
    signal_power = mean(clean.^2);
    error_power  = mean((enhanced - clean).^2) + eps;
    score = 10 * log10(signal_power / error_power);
end