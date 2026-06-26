function hIR = hIRgen(fs, c, rcWall, impluseLen, highpass, win, lroom, rmic, rs)
% impulse response
% Date: 2021.5.23
% Reference:
%       [1] J.B. Allen, D.A. Berkley,
%       Image method for efficiently simulating small-room acoustics,
%       J. Acoust. Soc. Am., vol. 65, pp. 943-950, 1979.

rirArg.fs = fs;                             % sampling frequency in Hz.
rirArg.c = c;                               % speed of sound in m/s.
rirArg.rcWall = rcWall;                     % T60 mode/s
rirArg.npts = impluseLen;                   % length of impulse response
rirArg.hPass = highpass;
rirArg.win = win;
rirArg.roomSize = lroom;                    % room size
rirArg.rmic = rmic;
rirArg.rs = rs;

nSrc = size(rs, 1);
nMic = size(rmic, 1);

hIR = zeros(impluseLen, nMic, nSrc);
for sIdx = 1:size(rs, 1)
    for mIdx = 1:size(rmic, 1)
        H = ImageRIR_Jin(rs(sIdx, :), rmic(mIdx, :), lroom, rirArg);
        [hIR(:, mIdx, sIdx)] = H*10;
    end
end
end

