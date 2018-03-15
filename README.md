# Statistical modeling
1. Statistical modeling process
 * Understand the problem
 * Plan & collect data
 * Explore data
 * Postulate model
 * Fit model
 * Check model
 * iterate (from 4-6)
 * use the model
# Bayesian modeling
1. Components of Bayesian models
 * likelihood: probabilistic model for the data. p(y|theta)
 * Prior: probability distribution that character the uncertainty with the parameter theta. p(theta)
 * posterior: p(theta|y) = likelihood * prior
2. Model specification
 * P(mu, sigma^2) = p(mu)p(sigma^2)
 * If you have the independent parameters (mu and sigma^2), you can generate the data by those parameters. This is the prior predicative distribution for data.
3. Posterior derivation
 * Two layers model:
 
		y | mu, simga^2 ~ N(mu, sigma^2) <br />
		mu ~ N(mu0, sigma0^2) <br />
		sigma^2 ~ IG(v, beta)
 * Three layers model:
		
		y | mu, simga^2 ~ N(mu, sigma^2) <br />
		mu | sigma^2 ~ N(mu0, sigma0^2/w0) <br />
		sigma^2 ~ IG(v, beta)
 * The chain rule of joint distribution of chain rule:
 	P(y1....yn,mu, sigma^2) = p(y1....yn|mu,sigma^2)P(mu|sigma^2)P(sigma^2)
							= Multiple(N(yi|mu, sigma^2)\* N(mu|mu0, sigma^2/w0)\*IG(sigma^2|mu0,beta0))
4. Non-conjugate models
 * We have the posterior distribution up to a normalizing constant, but we are unable to integrate it to obtain import quantities, such as posterior mean or probability intervals.
 * Some complicated model can't be written as standard form of distribution. This is why some sampling method are invited. MCMC, variational inference.r
# Monte Carlo Estimation
1. Monte Carlo integration:
	* Monte Carlo 方法通过使用多次抽样模拟的方法，通过中心极限定理来近似求出一个复杂的概率分布的指标， 比如均值，方差等等。
2. Monte Carlo error and marginalization
	* The variance of estimate is controlled in part of m(the sample size)
		a. theta_bar ~ N(E(theta),var(theta)/m) 
			* theta_bar: sample mean
		b. var_hat(theta) = 1/m * SUM(theta_i - theta_bar)^2
			* var_hat : sample variance
		c. sqrt(var_hat/m) = standard error
	* Example:
		* Y|phi ~ Bernoulli(10,phi) and phi ~ Beta(2,2)
		* simulate: 
			1. draw sample phi_i from Beta distribution by 2 and 2
			2. Given phi_i draw sample y_i from bernoulli(10, phi_i)
			3. We can get many pairs of (y_i, phi_i)
# Markov chains:
1. Conditional independence:
	* If previous state(t-1) can be observed,P(xt|xt....x1) would be p(xt|xt-1). 
	* Markov chain is based on the assumption of conditional independence.
2. Discrete Markov chain:
3. Random walk (continuous Markov chain):
	* Transition matrix of Random walk is a Normal distribution. mean is previous state and var is 1. 
	* Random walk don't have stationary distribution. A modification can make it get stationary state. mean of new transition normal distribution is phi*x(t-1). Here, phi belongs to range of (-1,1). After that, MC from stationary distribution to compute mean and std.
4. Transition matrix:
	* A matrix Q record the state transition probability. Row means t-1 and column means t.
	* N times state transition = Q^N
	* transition probability from state A to state -pc B can be found up from status^(B-A) directly.
5. Stationary distribution:
	* After many times (for example 100) state transition, the probability of state -pc would converge to a stationary distribution.
	* The stationary distribution of a chain is the initial state distribution for which performing a transition will not change the probability of ending up any given state. This is exactly how we are going to use Markov chains for Bayesian inference. In order to simulate from a complicated posterior distribution, we will set up and run a Markov chain whose stationary distribution is the posterior distribution.


## Metropolis-Hastings
Sometimes, we can't compute the normalizing constant easily or the itegral of posterior distribution is very difficult to compute. We need to find the other distribution(proposal distribution) to approximate real one, and sample from that approximate distribution.

1. Algorithm(MH)
	
	* MH method can be used in two case:
		- Independent MH, the proposed distribution q(x) must be as similar as p(x).
		- Dependent MH, next state is depending on the previous state.
		- In dependent MH, if q(s) is symmetric, item of q(x) can be canceld out from eqution of accept ratio of alpha.
	* Steps:
	
		- Select initial value theta0
		- for i=1,...,m repeat:
			+ Draw candidate from theta\* ~ q( theta\* | theta\_(i-1) )
			+ alpha = g( theta\* )q( theta\_(i-1) | theta\* ) / g( theta\_(i-1) )q( theta\* | theta\_(i-1) )
			+ alpha>=1, 
					
					accept theta* set theta_i <- theta*
			+ 1>=alpha>=0, 
					
					accept theta* and set theta_i <-theta* with prob.alpha
					reject theta* and set theta_i <- theta_(i-1) with prob.1-apha
2. Demonstration
		
		demonstration_question_solution.pdf
3. Random walk example
	
	Random walk MH is the MH method with proposed distribution of Normal distribution.
		
		the code can be found in course matrerial
4. Reading:
		
		Hoff: The Metropolis algorithm
		Gelman et al: Why does the Metropolis algorithm work?

## JAGS

## Gibbs Sampling
1. Multiple parameter sampling and full conditional distributions
2. Conditionally conjugate prior example with normal likelihood
3. Computing example with normal distribution

## Assessing Converge
1. Trace plots and autocorrelation
2. Multiple chains, burn-in and Gelman-Rubin diagnostic

## MCMC

