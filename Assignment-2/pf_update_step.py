import numpy as np
from map_measurement_loglik import map_measurement_loglik
# Particle Filter update step
#   Evaluates measurement log-likelihood for all particles,
#   and resamples, with probability proportional to their likelihood.
def pf_update_step(particles, meas, map_, sensor):
    # number of particles
    N = particles.shape[1]

    # this list will contain the (log of the) weights for all N particles
    log_weights = np.zeros(N) - 9999

    ## evaluate particle likelihood

    # For each particle:
    #     compute the (unnormalized) measurement log-likelihood
    #     store the log-likelihood of the j-th particle as weights(j)
    # Tip 1: use a simple for-loop to iterate over all N particles
    # Tip 2: use map_measurement_loglik

#########################
## YOUR_CODE_GOES_HERE ##
#########################

    ## resample particles 

    # We construct construct normalized probabilities from the list of
    #   log-likelihoods. Basically, we need to convert log-likelihoods to
    #   normal likelihoods, and then divide by their sum to obtain a
    #   normalized probability distribution (i.e. that sums to 1).
    #
    # Some background information on the following lines:
    # Before converting to normal probabilities, we perform a 'trick'
    # to ensure the the unnormalized probabilities do not become too small
    # to suffer from the rounding errors of the computer when using exp().
    # The trick is to multiply the probabilities by some rescaling factor,
    # which in log-space means adding or subtracting the log factor.
    # This scaling factor will be removed in the final normalization
    # step anyway.
    assert len(log_weights.shape) == 1
    probs = log_weights # start with the original log-likelihoods
    probs = probs - np.max(probs) # the "big trick"
    probs = np.exp(probs) # convert log-probabilities to probabilities
    probs = probs / np.sum(probs) # normalize

    # now probs should sum to one (with minimum tolerance for nasty
    # rounding errors).
    assert np.abs(np.sum(probs) - 1.) < 1e-10

    # Sample a new set of N particles (with replacement) from old set,
    # and call this set new_particles
    #
    # NOTE: you can use the np.random.choice function to sample with replacement.
#########################
## YOUR_CODE_GOES_HERE ##
#########################

    # new_particles should by a 3 x N matrix
    assert new_particles.shape == (3,N)

    return new_particles
