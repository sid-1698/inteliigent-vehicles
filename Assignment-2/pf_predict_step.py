import numpy as np

# Particle Filter predict step
# Input:
#   - particles: 3 x N matrix with N particles (as columns) of previous
#     time step
#   - control_input: a 2D vector with [forward velocity, steering angle]
#   - dt: time step duration (in seconds)
# Output:
#   - particles: 3 x N matrix with N updated particles for new timestep

#function particles = pf_predict_step(particles, control_input, dt)
def pf_predict_step(particles, control_input, dt):
    # number of particles
    N = particles.shape[1]
    
    # in this simple model there is only velocity and steering
    velocity = control_input[0]        # [   v   ] forward velocity
    steering_angle = control_input[1]  # [ omega ] steering angle

    # Here the prediction happens with the simple motion model,
    #  and the added Gaussian noise
    
    ## deterministic motion
    # First apply deterministic motion model:
    #    [  x_t  ]      [  x_t-1  ]    [ v * dt * sin(theta_t-1) ]
    #    [  y_t  ]  =   [  y_t-1  ] +  [ v * dt * cos(theta_t-1) ]
    #    [theta_t]      [theta_t-1]    [       omega * dt        ]
    # for i in range(1,N):
    #     particles[0,i] = particles[0,i-1] + velocity* dt * np.sin(particles[2,i-1])
    #     particles[1,i] = particles[1,i-1] + velocity* dt * np.cos(particles[2,i-1])
    #     particles[2,i] = particles[2,i-1] + steering_angle * dt
    
    x_t = np.array([velocity*dt*np.sin(particles[2,:])])
    y_t = np.array([velocity*dt*np.cos(particles[2,:])])
    theta_t = np.array([np.transpose(np.squeeze(np.ones([1,int(N)])*steering_angle*dt))])
    predict = np.array([x_t,y_t,theta_t])
    particles=particles+np.squeeze(predict)
#########################
## YOUR_CODE_GOES_HERE ##
#########################

    ## add noise
    # Second, add here some random noise to each particle.
    # You can use multivariate_normal to generate noise to add to the particles.
    # For each particle i, sample a noise vector
    #     \eta^{(i)} ~ N(0, Sigma_x) .
    # Note that with multivariate_normal you can sample all N noise vectors in one go.
    
    Sigma_x = np.diag([
        .1,  # std.dev. on x position is 0.2 meters
        .1,  # std.dev. on y position is 0.2 meters
        (2.*np.pi / 24.)**2  # std.dev. on angle is 1/24 of full 360 degrees circle
    ])
    # eta = np.zeros((3,N))
    # eta = np.random.multivariate_normal(np.array([0,0,0]),Sigma_x,N)
    # particles+= np.transpose(eta)
    
    for i in range(len(particles[0])):
         eta = np.random.multivariate_normal(np.array([0,0,0]),Sigma_x,1)
         eta = np.squeeze(eta)
         particles[:,i] = particles[:,i] + np.transpose(eta)
        
        

#########################
## YOUR_CODE_GOES_HERE ##
#########################

    # Since the dimension of each particle, i.e. orientation theta,
    # is on the circular domain, let's ensure all values remain on the
    # circular [0, 2*pi] domain.
    particles[2,:] = np.mod(particles[2,:], 2.*np.pi)
    return particles
