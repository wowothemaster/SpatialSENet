function [X_output_matrix,stft_parameters] = implement_stft(x_input_signal, frmsize)

% implementation the STFT to a single/muli-channel channel sinals.
%
%
% USAGE:
% [X_output_matrix,stft_parameters] = implement_sift_single(x_input_signal)
%
% Inputs summary:
% x_input_signal:    the input signal, can be single/muli-channel time domain signal
% ----------------------------------------------------------------
% Output:
% X_output_matrix:   the output signal in STFT domain
% stft_parameters:   parameters for STFT

%
% Authors: Gongping Huang
% Date: 13/04/2023
%-------------------------------------------------------------------------
%

X_signal            = x_input_signal;
M                   = length(X_signal(1,:));

% parameters for STFT: frame and window sizes

% frmsize             = 128*1;
noverlap            = 4;
winsize             = frmsize*noverlap;
nfbands             = winsize/2+1;
delay               = (noverlap-1)*frmsize;

% Window parameters-Kaiser Window
kwin                = kaiser(winsize, 1.9*pi);
kwsigma             = sqrt(sum(kwin.^2)/winsize*noverlap);
kwin                = kwin/kwsigma;

% reconstruct the noisy signal-just for simulation
K                   = length(X_signal(:,1));
nfrms               = ceil(K/frmsize);
Kfrm                = nfrms*frmsize;
if Kfrm > K
    X_signal(K+1:Kfrm,:)   = 0;
end

% create the needed buffers for FFT/IFFT
inBuf               = zeros(winsize,M);   % input buffer
inWinBuf            = zeros(M,winsize);   % input fft buffer

if M < 2
    X_output_matrix  = zeros(nfbands,nfrms);
else
    X_output_matrix  = zeros(M,nfbands,nfrms);
end

    
for nf = 1:nfrms
    
    if M < 2
     % 1) Construct STFT domain signal for single channel
        
        yfrm1               = X_signal((nf-1)*frmsize+1:nf*frmsize,1);
        inBuf(:,1)          = [inBuf(frmsize+1:end,1); yfrm1];
        inWin1              = fft(inBuf(:,1).*kwin);
        
        X_output_matrix(:,nf)   = inWin1(1:nfbands);
        
    else
    % 2) Construct STFT domain signal for mcirophone array
        
        for m =  M : -1 : 1
            % do STFT to multivhannel signal at MA
            yfrm1               = X_signal((nf-1)*frmsize+1:nf*frmsize,m);
            inBuf(:,m)          = [inBuf(frmsize+1:end,m); yfrm1];
            inWin1              = fft(inBuf(:,m).*kwin);
            inWinBuf(2:end,:)   = inWinBuf(1:end-1,:);
            inWinBuf(1,:)       = inWin1;
        end
        
        X_output_matrix(:,:,nf)         = inWinBuf(:,1:nfbands);
    end
    
end

% save these parameters for istft
stft_parameters.frmsize     = frmsize;
stft_parameters.noverlap    = noverlap;
stft_parameters.winsize     = winsize;
stft_parameters.nfbands     = nfbands;
stft_parameters.kwin        = kwin;

end
