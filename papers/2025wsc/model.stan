// description: Varying effects with MVN, texts as fixed effects.

data {
	int<lower=1> N;  // number of records
	int<lower=1> O;  // number of covariates
	int<lower=1> nStrata; // number of strata
	int<lower=1,upper=nStrata> strata[N]; // stratum indicator for each record
	int<lower=1> nTexts; // number of different source texts w/o period stratification
	int<lower=1,upper=nTexts> texts[N]; // indicator per record.
	int<lower=0,upper=1> continuous[N];   // binary outcome
	matrix[N,O] X;
}

parameters {
	vector[O] coefsM; // Global means
	matrix[nStrata, O] z; // Standard normal z-scores.
	vector<lower=1e-4>[O] sigma; // Standard deviations for each coefficient
	cholesky_factor_corr[O] L_Omega; // Cholesky factor of correlation matrix
	vector[nTexts] betaText;
}

transformed parameters {
	vector[N] rates;
	vector[N] logits;
	vector[N] Lp;  // Log probability for LOO
	matrix[nStrata, O] coefsStrata; // Stratum-specific coefficients

	// Non-centered parameterization with full covariance
	for (t in 1:nStrata) {
		coefsStrata[t] = coefsM' + (diag_pre_multiply(sigma, L_Omega) * z[t]')';
	}

	// Calculate rates and log probabilities
	for (n in 1:N) {
		logits[n] = (X[n] * coefsStrata[strata[n]]') + betaText[texts[n]];
		rates[n] = inv_logit( logits[n] );
		Lp[n] = bernoulli_lpmf(continuous[n] | rates[n]);
	}
}

model {
	coefsM ~ std_normal();
	sigma ~ exponential(5);
	L_Omega ~ lkj_corr_cholesky(2);
	to_vector(z) ~ std_normal();
	betaText ~ normal(0,0.5);
	// Likelihood
	target += sum(Lp);
}

generated quantities {
	int continuous_samples[N];
	matrix[O, O] Omega;    // Correlation matrix
	matrix[O, O] Sigma;    // Covariance matrix

	// Correlation and covariance matrices
	Omega = multiply_lower_tri_self_transpose(L_Omega);
	Sigma = quad_form_diag(Omega, sigma);

	// Samples for PPC.
	for (n in 1:N) {
		continuous_samples[n] = bernoulli_rng(rates[n]);
	}
}
