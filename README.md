# comp-astro-project
This is the final project for computational astrophysics, which will focus on simulating galaxy dynamics using (1) the Leapfrog algorithm and (2) tree code. We will compare the computational efficiency of both algorithms and the maximum number of particles that can be evolved within a given time frame.

We reproduce many of the results from Toomre and Toomre 1972 (https://ui.adsabs.harvard.edu/abs/1972ApJ...178..623T/abstract).

The milestones for this project are:
-[ ] Set up the initial particle distribution and free parameters for each galaxy. We start with two simple cases: an elliptical galaxy that follows a r^(1/4) spherical mass distribution, and a flat disk galaxy that follows an exponential mass profile. Later, we generalize to allow any Sersic profile and oblateness.

-[ ] Implement interactions for the Toomre and Toomre scenario of massless outer particles affected by a single conglomerate mass at the center of each galaxy, and the more realistic interaction of a collection of rotationally or dispersion supported massive particles.

-[ ] Implement a leapfrog algorithm to evolve the forces, positions, and velocities of the particles over time. This will closely resemble Homework 4's code.

-[ ] Research tree codes much like that used by GADGET to see how many more test particles can be supported. Repeat above evolution.

-[ ] Reproduce the four basic simulations described by Toomre and Toomre.
\\\\
The four simulations follow the following requirements:

-[ ] 120 test particles surround a point mass representing one of the galaxies. Another galaxy is represented solely by a point mass
-[ ] The test particles are distributed in 5 annular rings of 12,18, 24, 30, and 36 particles respectively, at radii of 20, 30, 40, 50, and 60 percent of the distance of closest approach of the two galaxy point masses.
-[ ] $t=0$ represents the time of closest approach, with negative times representing the galaxies getting closer and positive times the galaxies growing farther apart.
-[ ] Time will be incremented by units of $10^8$ years.
-[ ] $R_min = 25$ kpc is used as the distance of closest approach, with the galaxy trajectories parabolic in nature.
-[ ] The heavier galaxy is $10^{11}$ solar masses.

Some specific scenarios we will try to replicate if time permits are:
-[ ] The bridge that connects M51 to NGC5195
-[ ] Arp 295's twin galaxies


