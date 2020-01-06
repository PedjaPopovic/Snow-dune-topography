import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.special import gammaincinv
from scipy.integrate import ode

"""
This code generates and plots the properties of the synthetic `snow dune' topography that is a sum of randomly sized and placed Gaussian mounds. 
The model is described in detail in the paper "Snow topography on undeformed Arctic sea ice captured by an idealized `snow dune' model." 
The code was tested on a 64-bit Mac using python 2.7.10.
"""

def Create_Snow_Dune_Topography(nx = 500, ny = -1, radius = 1., hm0 = 1, rho = 0.5, number_of_r_bins = 150, window_size = 5):
    
    """
    Create the synthetic "snow dune" topography.
    
    Parameters:
        
        nx: int
            Number of grid points in the x-direction
        ny: int
            Number of grid points in the y-direction. If ny < 0 then ny = nx.
        radius: float
            Mean mound radius, r0.
        hm0: float
            Mean mound height.
        rho: float
            Density of mounds.
        number_of_r_bins: int
            Number of bins in which to place mound radii.
        window_size: float
            Cutoff parameter for placing mounds. 
            
    Outputs:
        
        ice_topo: 2d numpy array
            The synthetic "snow dune" topography.
    """
    
    if ny < 0: ny = nx
    
    ice_topo = np.zeros([nx,ny])
    N = np.ceil(nx*ny/radius**2 * rho).astype(int)
    r0 = np.random.exponential(radius,N)

    bins = np.linspace(np.min(r0),np.max(r0),number_of_r_bins+1)
    r0_bins = np.zeros(number_of_r_bins)
    r0_N = np.zeros(number_of_r_bins).astype(int)

    for i in range(1,number_of_r_bins):
        loc = (r0 >= bins[i-1]) & (r0 < bins[i])
        r0_bins[i] = np.mean(r0[loc])
        r0_N[i] = np.sum(loc)

    r0_bins = r0_bins[r0_N>0] 
    r0_N = r0_N[r0_N>0] 

    for i in range(len(r0_bins)):
        r = r0_bins[i]
        h0 = hm0 * r / radius
        cov = np.eye(2)
        cov *= r**2

        rv = st.multivariate_normal([0,0], cov) 

        x0 = np.random.choice(np.arange(-nx/2,nx/2),r0_N[i])
        y0 = np.random.choice(np.arange(-ny/2,ny/2),r0_N[i])

        x = np.arange(-np.ceil(r*window_size).astype(int),np.ceil(r*window_size).astype(int)+1)
        y = x.copy()
        X,Y = np.meshgrid(x,y)
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X; pos[:, :, 1] = Y

        G = rv.pdf(pos) * 2 * np.pi * np.sqrt(np.linalg.det(cov)) * h0

        for j in range(r0_N[i]):
            loc_x = ((pos[:,:,0]+x0[j]) % nx).astype(int)
            loc_y = ((pos[:,:,1]+y0[j]) % ny).astype(int)

            ice_topo[loc_x,loc_y] += G
            
    return ice_topo
    
    
def find_ponds(ice_topo,p,dh_rel = 0.001):
    """
    Find the cutting plane of a topography for a given pond coverage fraction.
    
    Parameters:
        
        ice_topo: 2d numpy array
            Topography to consider.
        p: float
            Desired pond coverage fraction.
        dh_rel: float
            Increment used to search for the cutting plane.
            
    Outputs:
        
        ponds: boolean 2d array
            Binary array where True stands for regions of the topography lower than the cutting plane height.        
        h: float
            Height of the cutting plane.  
              
    """
    
    dh = dh_rel*(np.max(ice_topo) - np.min(ice_topo))
    
    h = np.max(ice_topo)
    ponds = ice_topo<h
    ph = np.mean(ponds)
    
    while ph > p:
        h-=dh
        ponds = ice_topo<h
        ph = np.mean(ponds)
    
    print('Pond coverage fraction = ' + str(ph) )
    
    return ponds, h
    
def find_moments(m_max,ice_topo):
    
    """
    Find moments of a topography.
    
    Parameters:
        
        m_max: int
            Maximum moment order to consider.
        ice_topo: 2d numpy array
            Topography to consider.
            
    Outputs:
        
        moment_order: list of ints
            Moment orders from 1 to m_max.        
        topo_moments: 1d array
            Moments of the topography, <ice_topo**m>.  
        root_topo_moments: 1d array
            Root-moments of the topography, <ice_topo**m>**(1/m)
              
    """
    
    topo_moments = np.zeros(m_max)  
    root_topo_moments = np.zeros(m_max)  
    moment_order = range(1,m_max+1)
    for m in moment_order:
                
        topo_moments[m-1] = np.mean(ice_topo**m)
        root_topo_moments[m-1] =  topo_moments[m-1]**(1./m)
    
    return moment_order, topo_moments, root_topo_moments

def center(Im):
    """
    Shift an image to place corners into center.
    
    Parameters:
        
        Im: 2d array
            Image to center.
            
    Outputs:
        
        Im_c: 2d array
            Centered image.                      
    """
    
    xlen = np.shape(Im)[1]
    ylen = np.shape(Im)[0]  
    Im_c = np.roll(np.roll(Im,xlen/2,axis = 1),ylen/2,axis = 0)
    
    return Im_c 

def correlation_function(ice_topo):
    
    """
    Find correlation function of the topography.
    
    Parameters:
        
        ice_topo: 2d array
           Topography to consider.
            
    Outputs:
        
        r: 1d array
            Distances considered.
        C_av: 1d array
            Radialy averaged topography correlation function evaluated at r.   
                               
    """
    
    F = np.fft.fft2(ice_topo)
    I = np.abs(np.fft.ifft2(F*np.conj(F)))
    
    xlen = np.shape(I)[1]
    ylen = np.shape(I)[0]

    
    A = xlen*ylen
    
    C = (center(I) - np.mean(ice_topo)**2 * A)/(A * np.std(ice_topo)**2)
    
    lx,ly = np.shape(C)
    r = np.arange(lx/2); phis = np.linspace(0,np.pi*2, 100)
    
    C_av = np.zeros(len(r))
    for phi in phis:
        
        x = lx/2+(r * np.sin(phi)).astype(int)
        y = ly/2+(r * np.cos(phi)).astype(int)
        C_av += C[x,y]
        
    C_av /= len(phis)
    
    return r,C_av

def f_stageI(t,y,params):
    
    """
    Right hand side of the differential equation for calculating stage I of 
    pond and water level evolution, dp/dt = f1(p,w), dw/dt = f2(p,w), 
    given by Eqs. 17 and 18 (if params['method'] == 'simple') 
    or Eqs. 18 and 19 (if params['method'] == 'complex') of the paper.
    
    Parameters:
        
        t: float
           Time.
        y: array of length 2
           Pond coverage and water level, (p,w).
        params: dict
            Dictionary of parameters that enter the equation. Parameters that enter
            this dictionary are explained below in the main code.
            
    Outputs:
        
        f: array of length 2
            Rate of change of pond coverage and water level, (dp/dt, dw/dt)
                               
    """
    
    hs = params['dhs_dt']
    hps = params['dhps_dt']
    hpi = params['dhpi_dt']
    
    rho_w = params['rho_w']
    rho_i = params['rho_i']
    rho_s = params['rho_s']
    
    L = params['L']
    Q = params['Q']
    pc = params['pc']
    beta = params['beta']
        
    pi = params['pi']
    
    ri = rho_i/rho_w
    rs = rho_s/rho_i
    
    f_dist = params['f_dist']
    method = params['method']
    p = y[0]; w = y[1]
    
    if method == 'simple':
        dwdt = (ri*(1-p)*hs  - Q*np.max([0,p/pc-1])**beta / L**2 / rs)/(1./rs-1+p) 
    if method == 'complex':
        dwdt = (ri*(1-p)*hs - (1-ri)* ( hps*p + pi*(hpi/rs - hps) ) - Q*np.max([0,p/pc-1])**beta / L**2 / rs  )/(1./rs-1+p)
    dpdt = f_dist(w+hs*t) * (dwdt + hs)
    
    return np.array([dpdt,dwdt])
     
def stageI_analytical(params):
    
    """
    Calculate pond coverage and water level evolution during stage I
    using Eqs. 17 and 18 (if params['method'] == 'simple') 
    or Eqs. 18 and 19 (if params['method'] == 'complex') of the paper.
    
    Parameters:
        
        params: dict
            Dictionary of parameters that enter the equation. Parameters that enter
            this dictionary are explained below in the main code.
            
    Outputs:
        
        t: 1d array
            Time from 0 to params['tmax'] in increments of params['dt'].
        sol: array of shape (len(t),2)
            Pond coverage and water level as a function of time, (p(t),w(t))
                               
    """
    
    t1 = params['tmax']; dt = params['dt']
    y0, t0 = np.array([0,0]), 0
    
    r = ode(f_stageI).set_integrator('vode', method='bdf')
    r.set_initial_value(y0, t0).set_f_params(params)
    
    t = np.array([t0]); sol = y0.copy()
    while r.successful() and r.t < t1:
        t = np.hstack([t,r.t+dt])
        sol = np.vstack([sol,r.integrate(r.t+dt)])
        
    return t, sol

def stageI_numerical(ice_topo,params):
    
    """
    Numerically solve the full 2d model of stage I pond and water level 
    evolution on a prescribed topography.
    
    Parameters:
        
        ice_topo: 2d array
            Topography on which ponds are evolving.
            
        params: dict
            Dictionary of parameters that enter the equation. Parameters that enter
            this dictionary are explained below in the main code.
            
    Outputs:

        t_evol: 1d array
            Time from 0 to params['tmax'] in increments of params['dt'].
        p_evol: 1d array
            Pond coverage as a function of time, p(t).
        w_evol: 1d array
            Water level as a function of time, w(t).
                               
    """
    
    mpp = params['mpp']
    dt = params['dt']
    tmax = params['tmax']
    
    dhs_dt = params['dhs_dt']
    dhps_dt = params['dhps_dt']
    dhi_dt = params['dhi_dt']
    dhpi_dt = params['dhpi_dt']
    
    rho_w = params['rho_w']
    rho_i = params['rho_i']
    rho_s = params['rho_s']
    
    Q = params['Q']
    pc = params['pc']
    beta = params['beta']
    
    snow = ice_topo > 0
    water_volume = 0.
    
    w0 = 0.
    t = 0.
    
    t_evol = [0.]
    p_evol = [0.]
    w_evol = [0.]
    while t < tmax:
        
        snow = ice_topo > 0
        ice = ice_topo <= 0
        pond = ice_topo < w0
        bare =  ice_topo >= w0
        
        sb = snow & bare
        ib = ice & bare
        sp = snow & pond
        ip = ice & pond
        
        p = np.mean(pond)
        t += dt
            
        ice_topo[sb] -= dt * dhs_dt
        ice_topo[ib] -= dt * dhi_dt
        ice_topo[sp] -= dt * dhps_dt
        ice_topo[ip] -= dt * dhpi_dt
        
        water_volume += dt * ( rho_s * ( dhs_dt * np.sum(sb) + dhps_dt * np.sum(sp) ) + rho_i * ( dhi_dt * np.sum(ib) + dhpi_dt * np.sum(ip) ) ) * mpp**2 / rho_w
        water_volume -= dt * Q * np.max([0,p/pc-1])**beta
        
        ip_area = np.sum(ip)*mpp**2
        sp_area = np.sum(sp)*mpp**2
        sb_area = np.sum(sb)*mpp**2
        ip_V = np.sum(ice_topo[ip])*mpp**2
        sp_V = np.sum(ice_topo[sp])*mpp**2
                
        w0 = ( water_volume + ip_V + rho_s/rho_i * sp_V ) / ( ip_area + sp_area + (rho_i - rho_s)/rho_i * sb_area)
        
        t_evol.append(t)
        p_evol.append(p) 
        w_evol.append(w0)
                   
    t_evol = np.array(t_evol)
    p_evol = np.array(p_evol)
    w_evol = np.array(w_evol)
    
    return t_evol, p_evol, w_evol

def pond_development_diagram(p_star = 0.01):
    
    """
    Find the pond development diagram described in Fig. 6 of the paper.
    
    Parameters:
        
        p_star: float
            Threshold pond coverage.
            
    Outputs:

        sigma: 1d array
            Values of non-dimensional snow roughness considered.
        omega: 1d array
            Corresponding non-dimansional water level for which p < p_star by the end of stage I.                              
    """
    
    sigma = np.linspace(0.01,1.5,100)
    omega = sigma**2 * gammaincinv(1./sigma**2,p_star)
    return sigma, omega

    
# ----------------------- Defining parameters --------------------------

nx = 500                # Number of grid points in the x-direction
ny = -1                 # Number of grid points in the y-direction. ny = -1 means that ny = nx
radius = 1.             # Mean mound radius, r0
hm0 = 0.03              # Mean mound height, hm0
rho = 0.2               # Density of mounds

max_moment = 40         # Maximum moment of the topography to calculate

day = 3600.*24          # Number of seconds in a day
hour = 3600.            # Number of seconds in an hour
dt = 0.5 * hour         # Increment of time in seconds for finding pond evolution during stage I 
tmax = 15*day           # Maximum time in seconds for calculating stage I pond evolution

rho_i = 900.            # Density of ice
rho_w = 1000.           # Density of water
rho_s = 360.            # Density of snow
Lm = 334000.            # Latent heat of melting in J/kg

F = 272.                # Shortwave solar radiation flux in W/m2
Fr = -26.               # Sum of longwave, latent, and sensible heat fluxes in W/m2

Q0 = 0.                 # Maximum drainage rate, Q0, in m3/s
pc = 0.35               # Percolation threshold. Relevant only if Q0 > 0.
Domain_width = 1500.    # Width of the domain in meters. Relevant only if Q0 > 0.
beta =  0.000001        # Exponent that relates drainage rate to deviation of pond coverage from the percolation threshold, Q = Q0 * np.max([0,p/pc-1])**beta. 
                        # Set beta close to 0 to approximate Heavisade step function (no drainage for p<pc and Q = Q0 for p>pc). If beta = 0, numerical problems arise.

mpp = Domain_width / nx # Meters per pixel.

a_s = 0.7               # Bare snow albedo.
a_i = 0.55              # Bare ice albedo.
a_ps = 0.25             # Ponded snow albedo.
a_pi = 0.25             # Ponded ice albedo.

dhs_dt = ((1-a_s)*F + Fr)/(Lm*rho_s)        # Bare snow melt rate.
dhps_dt = ((1-a_ps)*F + Fr)/(Lm*rho_s)      # Ponded snow melt rate.
dhi_dt = ((1-a_i)*F + Fr)/(Lm*rho_i)        # Bare ice melt rate.
dhpi_dt = ((1-a_pi)*F + Fr)/(Lm*rho_i)      # Ponded ice melt rate.

params = {'dhs_dt': dhs_dt, 'dhps_dt': dhps_dt, 'dhi_dt': dhi_dt, 'dhpi_dt': dhpi_dt, # Parameter dictionary used to calculate stage I of pond evolution in both the analytical
          'rho_w': rho_w, 'rho_i': rho_i, 'rho_s': rho_s, 'pi': 0.,'L': Domain_width, # model and the full 2d numerical model. The dictionary here is missing two inputs, method and f_dist, 
          'Q': Q0, 'pc': pc, 'beta':  beta, 'dt': dt, 'tmax': tmax, 'mpp':mpp}        # that are used for analytical estimate and are defined below. method ('simple' or 'complex') determines 
                                                                                      # the differential equations used to estimate pond evloution analyticaly ('simple' - first order estimate, 
                                                                                      # Eqs. 17 and 18 of the paper; 'complex' - second order estimate, Eqs. 18 and 19 of the paper). f_dist is  
                                                                                      # the snow depth distribution estimated based on the mean and variance of the generated synthetic `snow dune' topography. 

# ---------------------- Generate "snow dune" topography ---------------

ice_topo = Create_Snow_Dune_Topography(nx = nx, ny = ny, radius = radius, hm0 = hm0, rho = rho)
ponds, h = find_ponds(ice_topo,p = 0.45)


# --------------------- Find moments of the topography -----------------

n_trials = 10
moments_normalized_array = np.zeros([max_moment, n_trials])
for j in range(n_trials):
    
    topo = Create_Snow_Dune_Topography(nx = nx, ny = ny, radius = radius, hm0 = hm0, rho = rho)
    moment_order, moments, moments_normalized = find_moments(m_max = max_moment,ice_topo = topo)
    moments_normalized_array[:,j] = moments_normalized
    
moments_normalized_mean = np.mean(moments_normalized_array, axis = 1)
moments_normalized_std = np.std(moments_normalized_array, axis = 1)



# ---------- Find correlation function of the topography ---------------

r_corr,Corr = correlation_function(ice_topo)

# ------------------- Stage I of pond evolution ------------------------

sigma_h = np.std(ice_topo)
k = (np.mean(ice_topo)/np.std(ice_topo))**2

shape_param = k
scale_param = sigma_h / np.sqrt(k)
rv = st.gamma(loc = 0, a = shape_param, scale = scale_param)
f_dist = rv.pdf

params['f_dist'] = f_dist

params['method'] = 'complex'
t_complex, sol_complex = stageI_analytical(params)
p_complex = sol_complex[:,0]
w_complex = sol_complex[:,1]

params['method'] = 'simple'
t_simple, sol_simple = stageI_analytical(params)
p_simple = sol_simple[:,0]
w_simple = sol_simple[:,1]

t_evol, p_evol, w_evol = stageI_numerical(ice_topo,params)


# --------------------- Diagram of pond development --------------------

sigma_1, omega_1 = pond_development_diagram(p_star = 0.01)
sigma_2, omega_2 = pond_development_diagram(p_star = pc)

# ------------------------------ Plots ---------------------------------


plt.figure(1)
plt.clf()

plt.subplot(1,2,1)
plt.imshow(ice_topo)
plt.title('Synthetic snow dune topography')

plt.subplot(1,2,2)
plt.imshow(ponds)
plt.title('Ponds on synthetic snow dune topography')


plt.figure(2)
plt.clf()

plt.subplot(1,2,1)
plt.plot(moment_order,moments_normalized_mean,'r--',lw = 2)
plt.fill_between(moment_order, moments_normalized_mean - moments_normalized_std, moments_normalized_mean + moments_normalized_std,facecolor='r',alpha = 0.3)

plt.xlabel('Moment order')
plt.ylabel(r'$\langle h^n \rangle^{1/n}[m]$',fontsize = 20)
plt.legend(['Mean over ensemble', 'Spread over ensemble (1 stdev)'], frameon = False, loc = 2,fontsize = 12)
plt.title('Snow dune topography root-moments')

plt.subplot(1,2,2)

plt.plot(r_corr*mpp,Corr,'r-',lw = 3)
plt.plot([0,r_corr.max()*mpp],[0,0],'k--',lw = 2)

plt.xlim(0,100*radius)

plt.xlabel('Distance [m]')
plt.ylabel('Correlation function')
plt.title('Snow dune topography correlaiton function')


plt.figure(3)
plt.clf()

plt.subplot(1,2,1)
plt.plot(t_evol/day,p_evol, lw=3)
plt.plot(t_simple/day,p_simple, 'r:', lw=2)
plt.plot(t_complex/day,p_complex, 'r--', lw=2)
plt.legend(['Numerical 2d model', 'First order analytical estimate', 'Second order analytical estimate'], frameon = False, loc = 4,fontsize = 12)

plt.title('Pond coverage evolution')
plt.xlabel('Time[days]')
plt.ylabel('p')

plt.subplot(1,2,2)
plt.plot(t_evol/day,w_evol, lw=3)
plt.plot(t_simple/day,w_simple, 'r:', lw=2)
plt.plot(t_complex/day,w_complex, 'r--', lw=2)

plt.title('Water level evolution')
plt.xlabel('Time[days]')
plt.ylabel('w[m]')
plt.legend(['Numerical 2d model', 'First order analytical estimate', 'Second order analytical estimate'], frameon = False, loc = 4)

plt.figure(4)
plt.clf()

plt.plot(sigma_1, omega_1,'k--',lw = 2)
plt.plot(sigma_2, omega_2,'k--',lw = 2)
plt.fill_between(np.hstack([0,sigma_1]), np.hstack([1,omega_1]), np.hstack([1,omega_2]), facecolor='b',alpha = 0.1)
plt.fill_between(np.hstack([0,sigma_2]), np.hstack([1,omega_2]), 2*np.ones(len(sigma_2)+1), facecolor='b',alpha = 0.3)

plt.xlabel(r'$\Sigma$',fontsize = 18)
plt.ylabel(r'$\omega$',fontsize = 24)
plt.title('Diagram of pond development')

plt.ylim(0,1.1)
plt.xlim(0,sigma_1.max())


plt.show()