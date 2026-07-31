Paper 1 — Causal Inference Models
Model	Purpose
Log-normal structural causal model (twin-model)	Core causal framework: Z→T→Y with shared parameters θ
Log-space OLS (no-intercept variant)	Fits exact power-law structural equations (a, b, c)
Log-space OLS (with-intercept variant)	Alternative fit including seasonal/month control
Naive linear regression (log Y ~ log T)	Confounded baseline for comparison
Alternative causal graph model (T→Z→Y ordering)	Non-identifiability demonstration
Two counterfactual models with identical marginals, different noise structure	Counterfactual non-identifiability demonstration
Paper 2 — Random Fourier Features Models
Model	Purpose
MLPRegressor — raw features (baseline)	Comparison baseline
MLPRegressor — RFF-preprocessed (Gaussian kernel)	Best-performing RFF variant
MLPRegressor — RFF-preprocessed (Laplacian kernel)	Kernel comparison
MLPRegressor — RFF-preprocessed (Cauchy kernel)	Kernel comparison
MLPRegressor — MinMaxScaler [0,1] variant	Alternative scaling comparison
Hand-coded ReLU MLP (manual backprop, NumPy)	NTK boundedness/Theorem 1 verification
Zero-bias ReLU MLP (homogeneity test network)	Appendix A lemma verification (exact homogeneity)
Paper 3 — Forecast-Then-Optimize Models

Base forecasters:

Model	SMAPE (%)
Naive Seasonal	26.65
Linear Trend (+ one-hot day-of-week)	12.55
MLP on lag features	18.10
TCN-style (dilated causal convolution)	34.45
DLinear (SMA trend/seasonal decomposition)	36.70
NLinear (last-value normalization)	24.18
RevIN + Linear (instance normalization)	19.29
Plain window-linear (no normalization)	27.34
EMA-decomposition (xPatch-style)	36.69

Ensembles/post-processing:

Method	SMAPE (%)
Mean Ensemble	16.37
Median Ensemble	15.52
Weighted-Median Ensemble	15.52
Quantile-0.1 Ensemble	26.09
Quantile-0.25 Ensemble	21.10
Quantile-0.5 Ensemble	15.52
Quantile-0.75 Ensemble	12.45
Quantile-0.9 Ensemble (best overall)	11.68
Dynamic Weighting Ensemble (sliding window)	15.02
Static BMA-like (inverse-error weighted)	17.13

Meta-learners:

Method	SMAPE (%)
Direct Loss Estimator (DLE / residual learning)	44.01
FFORMA-style instance-specific weighting	17.46
Conformal Prediction Interval	radius ±18,928 (100% coverage vs. 90% target)

Total: 6 causal models + 7 RFF/kernel models + 9 base forecasters + 10 ensemble/meta-learner methods = 32 distinct models/techniques implemented and run across the full project.
