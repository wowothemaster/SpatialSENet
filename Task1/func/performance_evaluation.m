function [overall_PESQ_origin,overall_PESQ_enhanced,overall_STOI_origin,overall_STOI_enhanced,overall_SDR_origin,overall_SDR_enhanced] = performance_evaluation (Sig_ref, Sig_observed, Sig_der, Fs)
%% all path in the current folder are included.
warning('OFF')
addpath(genpath(pwd))
%% 1) Overall Performance Evaluation %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% align amplitude
Sig_ref                 = 0.5*Sig_ref./max(max(Sig_ref));
Sig_observed            = 0.5*Sig_observed./max(max(Sig_observed));
Sig_der                 = 0.5*Sig_der./max(Sig_der);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % 1) Overall Performance Evaluation

source1          = evaluate_score(Sig_ref, Sig_observed, Fs);
overall_PESQ_origin     = source1(1);
overall_STOI_origin     = source1(2);
overall_SDR_origin      = source1(3);

source2         = evaluate_score(Sig_ref, Sig_der, Fs);
overall_PESQ_enhanced   = source2(1);
overall_STOI_enhanced   = source2(2);
overall_SDR_enhanced    = source2(3);