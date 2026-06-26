function [output_signal, delay] = implement_istft(Z_output,stft_parameters)


% transform the input signal in the STFT domain of size (K) X (T_N) back to
% the time domain.
%
%
% USAGE:
% [output_signal, delay] = implement_istft(Z_output,stft_parameters)

%
% Inputs summary:
% Z_output:         the input STFT coefficients of size (K) X (T_N) 
% stft_parameters:  parameters for STFT
% ----------------------------------------------------------------
% Output:
% output_signal:    the single-channel output signal in the time domain
% delay:            delay casued by overlap

%
% Authors: Gongping Huang
% Date: 13/04/2023
%-------------------------------------------------------------------------
%

% parameters for STFT: frame and window sizes

frmsize             = stft_parameters.frmsize;
noverlap            = stft_parameters.noverlap;
winsize             = stft_parameters.winsize;
nfbands             = stft_parameters.nfbands;
delay               = (noverlap-1)*frmsize;
kwin                = stft_parameters.kwin;

% 1)reconstruct the noisy signal-just for simulation
nfrms               = length(Z_output(1,:));

% 1)Create the needed buffers for FFT/IFFT
outWin              = zeros(winsize,1);   % output analysis window
outOAWin            = zeros(winsize+frmsize,1);%output overlap-add window

% 1-10)Create the needed buffers for finally output
z_output            = zeros(nfrms*frmsize,1);

%% %speech enhancement process

for nf = 1:nfrms

    outWin(1 :nfbands)      = Z_output(:,nf);
    
    % 3-1) noisy signal after beamforming
    outWin(nfbands+1:end)   = conj(flipud(outWin(2:nfbands-1)));
    % inverse fft
    outWin = real(ifft(outWin));
    
    % overlap add
    outOAWin(frmsize+1:end) = outOAWin(frmsize+1:end)+ kwin.*outWin;
    outOAWin(1:winsize)     = outOAWin(frmsize+1:end);
    outOAWin(winsize+1:end) = 0;
    % copy to the output buffer
    outBuf  = (outOAWin(1:frmsize));
    z_output((nf-1)*frmsize+1 : nf*frmsize) = outBuf;
    
end
 
output_signal         = z_output(:);

end

