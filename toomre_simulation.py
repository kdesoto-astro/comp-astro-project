import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
#from joblib import Parallel, delayed
#import multiprocessing

G = 4492.481912367
TIME_RES = 1e-3

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
        self.dt = TIME_RES
    
    
class OuterSet:
    """
    Object to store each particle. Assumed to be massless.
    """
    def __init__(self, t_start):
        self.x = np.array([])
        self.y = np.array([])
        self.vx = np.array([])
        self.vy = np.array([])
        self.dt = TIME_RES
        self.t = t_start
    
    def append(self, x, y, vx, vy):
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)
        self.vx = np.append(self.vx, vx)
        self.vy = np.append(self.vy, vy)
    
    
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
        R_min_arr = [R_min*reduced_mass / m1, R_min*reduced_mass / m2]
        self.rmin = R_min
        for g_idx in [0,1]:
            #print(R_min[g_idx])
            k = G * reduced_mass
            true_anomaly0 = parabolic_true_anomaly(R_min_arr[g_idx], reduced_mass, t_start)
            #print("true anomaly", true_anomaly0)
            r = 2. * R_min_arr[g_idx] / (1. + np.cos(true_anomaly0))
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
                self.g1 = GalaxyCenter(m1, R_min_arr[g_idx], x, -y, vx, -vy, t_start)
            else:
                self.g2 = GalaxyCenter(m2, R_min_arr[g_idx], -x, y, -vx, vy, t_start)
        
        self.particles = OuterSet(t_start)
        for disc_r_frac in radius_to_particle_fraction:
            
            # initialize particles for each ring in disk
            R = disc_r_frac * R_min
            num_particles = int(np.round(self.N * radius_to_particle_fraction[disc_r_frac]))
            v_kepler =  np.sqrt(G * m1 / R)
            
            # evenly distribute particles in that ring
            theta = np.linspace(0., 2.*np.pi, num=num_particles+1)[:-1]
            x = R * np.cos(theta) + self.g1.x
            y = R * np.sin(theta) + self.g1.y
            if retrograde:
                vy = 1. * v_kepler * np.cos(theta) + self.g1.vy
                vx = -1. * v_kepler * np.sin(theta) + self.g1.vx
            else:
                vy = -1. * v_kepler * np.cos(theta) + self.g1.vy
                vx = 1. * v_kepler * np.sin(theta) + self.g1.vx
            self.particles.append(x, y, vx, vy)
    
    
    def leapfrog_kick(self, p):
        
        r_sq1 = (p.x - self.g1.x)**2 + (p.y - self.g1.y)**2
        r_sq2 = (p.x - self.g2.x)**2 + (p.y - self.g2.y)**2
        ax =  -G * self.g1.m * (p.x - self.g1.x) / r_sq1**(3/2.) \
            - G * self.g2.m * (p.x - self.g2.x) / r_sq2**(3/2.)
        ay =  -G * self.g1.m * (p.y - self.g1.y) / r_sq1**(3/2.) \
            - G * self.g2.m * (p.y - self.g2.y) / r_sq2**(3/2.)
        
        #print(self.g1.x, p.x, ax)
        p.vx += ax * p.dt / 2.
        p.vy += ay * p.dt / 2.

    def step_leapfrog(self):
        
        #kick step
        self.leapfrog_kick(self.particles)

        # drift step
        self.particles.x += self.particles.vx * self.particles.dt
        self.particles.y += self.particles.vy * self.particles.dt

        # kick step
        self.leapfrog_kick(self.particles)

        self.particles.t += self.particles.dt
        
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
        #plt.scatter([self.g1.x, self.g2.x], [self.g1.y, self.g2.y], s=1, c="k")

        
    def evolve(self, t_final, img_num):
        while self.g1.t < t_final:
            if np.round(self.g1.t) == self.g1.t:
                print(self.g1.t)
            self.step_leapfrog()
        #print(self.g1.x, self.g1.y, self.g1.vx, self.g1.vy)
        """
        plt.xlim(-50., 50.)
        plt.ylim(-100., 100.)
        plt.savefig("figs/time_%.03e.png" % self.g1.t)
        #plt.ylim(-50., 50.)
        plt.close()
        """
        
        # plot particle positions
        plt.scatter(self.particles.x, self.particles.y, s=10, facecolors='none', edgecolors='k')
        plt.scatter([self.g1.x, self.g2.x], [self.g1.y, self.g2.y], s=20, c="k")
        plt.title("t = %01f" % self.g1.t)
        plt.xlim(-50., 100.)
        plt.ylim(-50., 100.)
        plt.savefig("figs/direct/toomre_a_%d.png" % img_num)
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


def make_video_from_images(image_folder, video_name):
    
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, fourcc, 50, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

def main():
    ts = ToomreSim(480, 1., 1., False, -10.)
    ct = 5000
    for et in np.linspace(-10., 10., num=500):
        ct += 1
        ts.evolve(et, ct)
    make_video_from_images("figs/direct", "toomre_direct_vid.avi")
    
main()
    
    