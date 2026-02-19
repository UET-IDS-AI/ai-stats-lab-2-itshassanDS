import numpy as np


# =========================
# Probability Functions
# =========================

def probability_union(pA, pB, pA_and_B):
    """
    P(A ∪ B) = P(A) + P(B) − P(A ∩ B)
    """
    return pA + pB - pA_and_B


def conditional_probability(pA_and_B, pB):
    """
    P(A | B) = P(A ∩ B) / P(B)
    """
    if pB == 0:
        raise ValueError("P(B) cannot be zero.")
    return pA_and_B / pB


def are_independent(pA, pB, pA_and_B, tol=1e-6):
    """
    A and B are independent if:
    P(A ∩ B) = P(A)P(B)
    """
    return abs(pA_and_B - (pA * pB)) < tol


def bayes_rule(pB_given_A, pA, pB):
    """
    Bayes' Rule:
    P(A | B) = [P(B | A) * P(A)] / P(B)
    """
    if pB == 0:
        raise ValueError("P(B) cannot be zero.")
    return (pB_given_A * pA) / pB


# =========================
# Bernoulli Distribution
# =========================

def bernoulli_pmf(x, p):
    """
    PMF of Bernoulli:
    P(X = x) = p^x (1-p)^(1-x)
    where x ∈ {0,1}
    """
    if x not in [0, 1]:
        return 0
    return p**x * (1 - p)**(1 - x)


def bernoulli_theta_analysis(theta_list, n_samples=100000):
    """
    For each theta:
    - Sample mean
    - Theoretical mean
    - Absolute mean error
    - Symmetry check (True if theta == 0.5)
    """
    results = []

    for theta in theta_list:
        samples = np.random.binomial(1, theta, n_samples)

        sample_mean = np.mean(samples)
        theoretical_mean = theta
        mean_error = abs(sample_mean - theoretical_mean)

        # Bernoulli is symmetric only when p = 0.5
        is_symmetric = abs(theta - 0.5) < 1e-6

        results.append((
            theta,
            sample_mean,
            theoretical_mean,
            is_symmetric
        ))

    return results


# =========================
# Uniform Distribution
# =========================

def uniform_histogram_analysis(a_list, b_list, n_samples=100000):
    """
    For each (a, b):
    - Sample mean
    - Theoretical mean = (a+b)/2
    - Mean error
    - Sample variance
    - Theoretical variance = (b-a)^2 / 12
    - Variance error
    """
    results = []

    for a, b in zip(a_list, b_list):
        samples = np.random.uniform(a, b, n_samples)

        sample_mean = np.mean(samples)
        theoretical_mean = (a + b) / 2
        mean_error = abs(sample_mean - theoretical_mean)

        sample_variance = np.var(samples)
        theoretical_variance = ((b - a) ** 2) / 12
        variance_error = abs(sample_variance - theoretical_variance)

        results.append((
            a,
            b,
            sample_mean,
            theoretical_mean,
            mean_error,
            sample_variance,
            theoretical_variance,
            variance_error
        ))

    return results


# =========================
# Normal Distribution
# =========================

def normal_histogram_analysis(mu_list, sigma_list, n_samples=100000):
    """
    For each (mu, sigma):
    - Sample mean
    - Theoretical mean = mu
    - Mean error
    - Sample variance
    - Theoretical variance = sigma^2
    - Variance error
    """
    results = []

    for mu, sigma in zip(mu_list, sigma_list):
        samples = np.random.normal(mu, sigma, n_samples)

        sample_mean = np.mean(samples)
        theoretical_mean = mu
        mean_error = abs(sample_mean - theoretical_mean)

        sample_variance = np.var(samples)
        theoretical_variance = sigma ** 2
        variance_error = abs(sample_variance - theoretical_variance)

        results.append((
            mu,
            sigma,
            sample_mean,
            theoretical_mean,
            mean_error,
            sample_variance,
            theoretical_variance,
            variance_error
        ))

    return results

