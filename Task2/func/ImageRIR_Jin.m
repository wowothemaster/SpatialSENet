function hIR = ImageRIR_Jin(rs, rmic, roomSize, param)

%---------------------------------------------------------------------------------------------------
% 
%   Description:
%       Calculate the ideal impulse response of an acoustic channel from the sources to the 
%   microphone (array) in a rectangular room.
%
%   Usage:
%       hIR = imageFIR(rs, rmic, lroom, rcwall, npts, highpass);
%
%   Inputs:
%       rs: coordinate of source signal (m);
%       rmic: coordinate of microphone (m); 
%       roomSize: size of rectangular room (longth, width, height);
%       rcWall: wall reflection coefficients or T60;
%       npts: length of impulse response;
%       hPass: flag of high passing the final impulse response; 
%
%   Outputs:
%       hIR: acoustic channel impulse response;
%
%   Author:
%       Jilu Jin, Center of Intelligent Acoustics and Immersive Communications,
%       Northwestern Polytechnical University,
%       charles.jilu.jin @ outlook.com    
%   
%   Reference:
%       [1] J.B. Allen, D.A. Berkley, 
%       Image method for efficiently simulating small-room acoustics, 
%       J. Acoust. Soc. Am., vol. 65, pp. 943-950, 1979.
%
%---------------------------------------------------------------------------------------------------

%% check function inputs

% limit the number of input arguments
narginchk(3, 4);

% normalize the speaker position, sensor position, and room size
rs = rs(:);
rmic = rmic(:);
roomSize = roomSize(:);

% check for input speaker position, sensor position, and room size
if length(rs) ~= 3
    error('Source position dimension error!');
end
if length(rmic) ~= 3
    error('Microphone position dimension error!');
end
if length(roomSize) ~= 3
    error('Room size dimension error!');
end

%% check parameters

% sampling rate and sound speed
if ~isfield(param, 'fs') 
    fs = 16e3;
else
    fs = param.fs;
end
if ~isfield(param, 'c')
    c = 340;
else
    c = param.c;
end

% reflection coefficients
if isfield(param, 'rcWall')
 	rcWall = param.rcWall;
    % single value -> T60 mode
    if length(rcWall) == 1
        if ~rcWall
            rcWall = zeros(3, 2);
        else
            roomVol = prod(roomSize);
            roomSurf = 2 * sum(roomVol ./ roomSize);
            % Sabine formula
            alpha = 24 * roomVol * log(10) / (c * roomSurf * rcWall);
            if (1 - alpha) < 0
                rcWall = zeros(3, 2);
                warning('No wall reflection!');
            else
                rcWall = ones(3, 2) * sqrt(1 - alpha);
            end
        end
    elseif sum(size(rcWall) == [3, 2]) ~= 2
        error('Reflection coefficients demension error.');
    end
    % check the value
    if any(rcWall(:) < 0) || any(rcWall(:) > 1)
        error('Invalid wall reflection coefficients!');
    end
else
	rcWall = 0.8 * ones(3, 2);
end

% length of impulse response
if isfield(param, 'npts')
 	npts = param.npts;
    if npts < 1
        error('Length of impulse response must be larger than 0.');
    end
else
	npts = 2^10;
end

%% advanced parameters

% highpass filter
if isfield(param, 'hPass')
 	hPass = param.hPass;
    if hPass ~= 0 && hPass ~= 1
        hPass = 0;
    end
else
	hPass = false;
end

% window
if isfield(param, 'win')
    win = param.win;
else
    win = 'hanning';
end

% window length
if isfield(param, 'wLen')
    wLen = param.wLen;
else
    wLen = 2 * round(4e-3*fs);
end
    
%% compute the impulse response

nVec = fix(2*npts./(roomSize/c*fs));

% (x, y, x) index
ix = (-nVec(1):nVec(1)).';
jy = (-nVec(2):nVec(2)).';
kz = (-nVec(3):nVec(3)).';

% compute the (x, y, z) position of image source
sx = ((-1).^ix)*rs(1) + 2*ceil(ix/2)*roomSize(1) - rmic(1);
sy = ((-1).^jy)*rs(2) + 2*ceil(jy/2)*roomSize(2) - rmic(2);
sz = ((-1).^kz)*rs(3) + 2*ceil(kz/2)*roomSize(3) - rmic(3);

% compute the distance of the receiver and image source
r = sx.^2 + permute(sy.^2, [3, 1, 2]) + permute(sz.^2, [3, 2, 1]);
r = sqrt(r);

% compute the reflection gain corresponding to (x, y, z) image
betax = (rcWall(1, 1).^abs(floor(ix/2))) .* (rcWall(1, 2).^abs(ceil(ix/2)));
betay = (rcWall(2, 1).^abs(floor(jy/2))) .* (rcWall(2, 2).^abs(ceil(jy/2)));
betaz = (rcWall(3, 1).^abs(floor(kz/2))) .* (rcWall(3, 2).^abs(ceil(kz/2)));
beta = betax .* permute(betay, [3, 1, 2]) .* permute(betaz, [3, 2, 1]);

% compute the time delay and propogation gain
td = r * fs / c;
gain = beta ./ (4*pi*r);

td = td(:);
gain = gain(:);

%%  Peterson's modification using hanning window

switch win
    case 'none'
    dLine = 0:npts-1;

    dIdx = td > npts + wLen;
    td(dIdx) = [];
    gain(dIdx) = [];

    hMat = gain .* sinc(td - dLine);
    hIR = sum(hMat);
    hIR = hIR(:);
    
    case 'hanning'
 	dIdx = td > npts + wLen;
 	td(dIdx) = [];
  	gain(dIdx) = [];
    
    dFix = td - floor(td);
    winMat = (1 - cos(2*pi*((1:wLen) - dFix)/wLen))/2;
    hSeg = sinc((1:wLen) - (dFix + wLen/2)) .* winMat .* gain;
    clear('winMat');
    
    hIR = zeros(npts + 2 * wLen, 1);
    for dIndex = 1:length(td)
        hStart = floor(td(dIndex));
    	hEnd = floor(td(dIndex)) + wLen - 1;
        hIR(hStart:hEnd) = hIR(hStart:hEnd) + hSeg(dIndex, :).';
    end
    hIR = hIR(wLen/2-1:wLen/2+npts-2);
end

%% highpass filtering

if hPass
    % Choice 1: 3rd order Elliptic digital filter
    % [hb, ha] = ellip(4, 0.1, 60, 2/50, 'high');
    % hc = filter(hb, ha, hc);

    % Choice 2: 3rd order IIR LPF from [J.Allen and D.Berkley, 1979].
    % hb = [1.0000; -1.9391; 0.9391];
    % ha = [1.0000; -1.8745; 0.8819];
    % hc = filter(hb, ha, hc);
    
    hb = [1.0000; -1.9391; 0.9391];
    ha = [1.0000; -1.8745; 0.8819];
    hIR = filter(hb, ha, hIR);
end