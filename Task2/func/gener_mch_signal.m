function [Xall,Xclean,Xearly,Xrever,Xinter] = gener_mch_signal(hIRs,hIRd,hIRe,hIRr,hIRi,Signal_para)

Lentime  = Signal_para.Lentime;
Signal_X = Signal_para.S_audio;
Signal_I = Signal_para.I_audio;
M        = Signal_para.M;

Signal_X = Signal_X(1:Lentime);

if length(Signal_I) > Lentime
    startIndex = randi(length(Signal_I) - Lentime + 1);
    Signal_I   = Signal_I(startIndex : startIndex + Lentime - 1);
else
    Signal_I   = [Signal_I; zeros(Lentime - length(Signal_I), 1)];
end

Signal_X = Signal_X / max(abs(Signal_X)) / 2;
Signal_I = Signal_I / max(abs(Signal_I)) / 2;

Xall     = zeros(length(Signal_X),M);
Xclean   = zeros(length(Signal_X),M);
Xearly   = zeros(length(Signal_X),M);
Xrever   = zeros(length(Signal_X),M);
Xinter   = zeros(length(Signal_I),M);

%% Generate signal

for M_idx = 1:M
    Xall(:, M_idx)   = Xall(:, M_idx)   + filter(hIRs(:, M_idx, 1), 1, Signal_X);
    Xclean(:, M_idx) = Xclean(:, M_idx) + filter(hIRd(:, M_idx, 1), 1, Signal_X);
    Xearly(:, M_idx) = Xearly(:, M_idx) + filter(hIRe(:, M_idx, 1), 1, Signal_X);
    Xrever(:, M_idx) = Xrever(:, M_idx) + filter(hIRr(:, M_idx, 1), 1, Signal_X);
    Xinter(:, M_idx) = Xinter(:, M_idx) + filter(hIRi(:, M_idx, 1), 1, Signal_I);
end


