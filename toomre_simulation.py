import numpy as np
import matplotlib.pyplot as plt

G = 4493.5032
    
def butcher_a_values():
    return np.array([[0., 0., 0., 0., 0., 0., 0.], \
            [0.2, 0., 0., 0., 0., 0., 0.], \
            [3./40, 9./40., 0., 0., 0., 0., 0.], \
            [44./45, -56./15, 32./9, 0., 0., 0., 0.], \
            [19372./6561, -25360./2187, 64448./6561, -212./729, 0., 0., 0.], \
            [9017./3168, -355./33, 46732./5247., 49./176, -5103./18656, 0., 0.], \
            [35./384, 0., 500./1113., 125./192, -2187./6784., 11./84, 0.]])

def butcher_b1_values():
    return np.array([35./384, 0., 500./1113., 125./192., -2187./6784, 11./84, 0.])

def butcher_b2_values():
    return np.array([5179./57600, 0., 7571./16695, 393./640, -92097./339200, 187./2100, 1./40])

def butcher_c_values():
    return np.array([0., 0.2, 0.3, 0.8, 8./9, 1., 1.])


def parabolic_true_anomaly(R_peri, reduced_mass, time_after_peri):
    """
    Calculates the true anomaly of parabolic orbit using hybrid root-finding.
    """
    #coeff = np.sqrt(2. * R_peri**3 / G / total_mass)
    coeff = np.sqrt(2. * R_peri**3/ (G * reduced_mass))
    theta_min = -np.pi
    theta_max = np.pi
    theta_curr = (theta_min + theta_max) / 2.
    ct = 0
    while True:
        ct += 1
        if ct > 1000:
            raise KeyboardInterrupt
        D = np.tan(theta_curr / 2.)
        f = coeff*(D + D**3/3.) - time_after_peri
        
        fprime = 0.5*coeff / np.cos(theta_curr / 2.)**4
        
        if np.abs(f) < 1e-12:
            if theta_curr < 0:
                return -1. * theta_curr
            return theta_curr
        
        #print(f)
        
        D_max = np.tan(theta_max / 2.)
        f_max = coeff*(D_max + D_max**3/3.) - time_after_peri

        if f_max * f > 0:
            theta_max = theta_curr
        else:
            theta_min = theta_curr
        
        theta_curr -= f/fprime
        if theta_curr > theta_max or theta_curr < theta_min:
            theta_curr = (theta_max + theta_min) / 2.
        

class GalaxyCenter:
    """
    Object to store a galactic center object.
    Has mass.
    """
    def __init__(self, m, rmin, x, y, vx, vy, t_start):
        self.m = m
        self.rmin = rmin
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.t = t_start
        self.dt = 1e-1
    
    
class OuterParticle:
    """
    Object to store each particle. Assumed to be massless.
    """
    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.dt = 1e-1
        self.t = -10. # in units of 1e8 years
    
    
class ToomreSim:
    """
    Framework for simulations 1-3 detailed in Toomre & Toomre. Uses
    adaptive-timestep RK4-5 to evolve the simulation.
    """
    def __init__(self, N, m1, m2, retrograde, t_start):
        self.N = N
        self.butcher_a = butcher_a_values()
        self.butcher_b1 = butcher_b1_values()
        self.butcher_b2 = butcher_b2_values()
        self.butcher_c = butcher_c_values()
        self.particles = []
        R_min = 25. # in kpc
        radius_to_particle_fraction = {
            0.2: 0.1,
            0.3: .15,
            0.4: 0.2,
            0.5: 0.25,
            0.6: 0.3
        }
        
        # initialize each galaxy parabolic trajectory
        reduced_mass = m1*m2/(m1+m2)
        R_min = [R_min*reduced_mass / m1, R_min*reduced_mass / m2]
        self.rmin = R_min
        for g_idx in [0,1]:
            #print(R_min[g_idx])
            k = G * reduced_mass
            true_anomaly0 = parabolic_true_anomaly(R_min[g_idx], reduced_mass, t_start)
            #print("true anomaly", true_anomaly0)
            r = 2. * R_min[g_idx] / (1. + np.cos(true_anomaly0))
            v = np.sqrt(2.*k/r)
            #print(180. * (np.pi - true_anomaly0) / np.pi)
            x = r*np.cos(np.pi - true_anomaly0) 
            y = r*np.sin(np.pi - true_anomaly0)
            #print(x, y)
            v_angle = np.pi / 2. - true_anomaly0 / 2.
            vx = - v * np.cos(v_angle)
            vy = - v * np.sin(v_angle)
            #print("v", vx, vy)
            
            if g_idx == 0:
                self.g1 = GalaxyCenter(m1, R_min[g_idx], x, y, vx, vy, t_start)
            else:
                self.g2 = GalaxyCenter(m2, R_min[g_idx], -x, -y, -vx, -vy, t_start)
        
        """
        for disc_r_frac in radius_to_particle_fraction:
            # initialize particles for each ring in disk
            R = disc_r_frac * R_min
            num_particles = np.round(self.N * radius_to_particle_fraction[disc_r_frac])
            dtheta = 2.*np.pi / num_particles
            theta = 0.
            v_kepler =  2.*G*m1 / R
            # evenly distribute particles in that ring
            for n in num_particles:
                theta += dtheta
                x = R * np.cos(theta)
                y = R * np.sin(theta)
                vy = -1. * v_kepler * np.cos(theta)
                vx = v_kepler * np.sin(theta)
                self.particles.append(OuterParticle(x, y, vx, vy))
        """
        
    def step_leapfrog(self):
        reduced_mass = self.g1.m*self.g2.m / (self.g1.m + self.g2.m)
        for g in (self.g1, self.g2):
            r_sq = g.x**2 + g.y**2 
            a =  -G * reduced_mass / r_sq**(3/2.)
            ax = a * g.x
            ay = a * g.y
            g.vx += ax * g.dt / 2.
            g.vy += ay * g.dt / 2.
            g.x += g.vx * g.dt
            g.y += g.vy * g.dt
            
            r_sq = g.x**2 + g.y**2 
            a =  -G * reduced_mass / r_sq**(3/2.)
            ax = a * g.x
            ay = a * g.y
            g.vx += ax * g.dt / 2.
            g.vy += ay * g.dt / 2.
            g.t += g.dt
            
            #print( - G * reduced_mass / np.sqrt(r_sq) + 0.5 * (g.vx**2 + g.vy**2)) # total energy
            
        plt.scatter([self.g1.x, self.g2.x], [self.g1.y, self.g2.y])

        
    def evolve(self, t_final):
        while self.g1.t < t_final:
            self.step_leapfrog()
        print(self.g1.x, self.g1.y, self.g1.vx, self.g1.vy)
        plt.xlim(-50., 50.)
        plt.ylim(-100., 100.)
        plt.savefig("figs/time_%.03e.png" % self.g1.t)
        #plt.ylim(-50., 50.)
        plt.close()
    
    def fourth_order_approx(self, particle, k_x, k_y, k_vx, k_vy):
        """
        Fourth order RK45 approximation formula
        """
        new_x = particle.x + self.dt*np.sum(self.butcher_b1*k_x)
        new_y = particle.y + self.dt*np.sum(self.butcher_b1*k_y)
        new_vx = particle.vx + self.dt*np.sum(self.butcher_b1*k_vx)
        new_vy = particle.vy + self.dt*np.sum(self.butcher_b1*k_vy)
        
        return new_x, new_y, new_vx, new_vy
    
    
    def fifth_order_approx(self, particle, k_x, k_y, k_vx, k_vy):
        """
        Fifth order RK45 approximation formula
        """
        new_x = particle.x + self.dt*np.sum(self.butcher_b2*k_x)
        new_y = particle.y + self.dt*np.sum(self.butcher_b2*k_y)
        new_vx = particle.vx + self.dt*np.sum(self.butcher_b2*k_vx)
        new_vy = particle.vy + self.dt*np.sum(self.butcher_b2*k_vy)
        
        return new_x, new_y, new_vx, new_vy

    
ts = ToomreSim(120, 1., 1., True, -10.)
ts.evolve(10.)
    
    