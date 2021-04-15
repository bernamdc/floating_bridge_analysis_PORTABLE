# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 12:08:57 2018

@author: bernardc

This script runs wind_field_3D.py function, tests, and hopefully validates it
To reduce the calculation time / memory, reduce: num_nodes, T, sample_freq.
"""

import numpy as np
import matplotlib.pyplot as plt
import math

# =============================================================================
import os
os.chdir("C:\Work\Bjornafjorden\Flytebru\Dynamic_Buckling\Python\Wind Field Script") # current directory

# =============================================================================
# Generating Node Coordinates. First, with Origin at circle center (0):
# =============================================================================
num_nodes = 6 # number of g_nodes where wind is generated
wind_dir = 90*2*np.pi/360 # 0 deg from North! 90 deg from West
zbridge = 16
R=5000      #*1000000 #m (horizontal radius of the bridge)
chord=5000 #m (straight distance from shore to shore)
sagitta=min((8*R+math.sqrt(64*R**2-16*chord**2))/8,
            (8*R-math.sqrt(64*R**2-16*chord**2))/8) 
            #Arch height = rise = sagitta
            #https://en.wikipedia.org/wiki/Circular_segment
            #R=h/2+c**2/(8H), solved for h
teta=2*math.asin(chord/2/R) #rad (angle from shore to shore)
startangle=-teta/2
numelem=num_nodes-1 #number of elements
deltateta=teta/numelem #each element's delta teta
nodes=np.array(list(range(num_nodes)))  #starting at 0
elemteta=nodes*deltateta+startangle
nodesxcoor0=R*np.sin(elemteta)
nodesycoor0=R*np.cos(elemteta)
nodeszcoord0=np.array([zbridge]*num_nodes)
nodescoor0=np.column_stack((nodesxcoor0,nodesycoor0,nodeszcoord0))
#...then with Origin at first node:
nodes=nodescoor0-nodescoor0[0]+[0,0,zbridge]
nodes_x = nodes[:,0]
nodes_y = nodes[:,1]
nodes_z = nodes[:,2]

rotation_matrix=np.array([[ np.cos(wind_dir) , -np.sin(wind_dir) , 0 ],
                          [ np.sin(wind_dir) ,  np.cos(wind_dir) , 0 ],
                          [ 0                ,  0                , 1 ]])
nodes = np.einsum('ni,ij->nj', nodes , rotation_matrix)

# Alternative node coordinates:
#nodes_x = np.linspace(0,5000,num_nodes)
#nodes_y = np.linspace(0,850,num_nodes)
#nodes_z = np.linspace(16,16,num_nodes)
#nodes_x = np.logspace(np.log10(150),np.log10(150),num_nodes)
#nodes_y = np.logspace(np.log10(1),np.log10(500),num_nodes)
#nodes_z = np.logspace(np.log10(16),np.log10(16),num_nodes)
#g_nodes = np.moveaxis(np.array([nodes_x, nodes_y, nodes_z]), -1, 0)

# =============================================================================
# Wind Parameters
# =============================================================================
V = 29.5 * ((nodes[:,2]/10)**0.127)

Au = 6.8
Av = 9.4
Aw = 9.4

Cux = 3. # ! Taken from paper: https://www.sciencedirect.com/science/article/pii/S0022460X04001373
Cvx = 3. # ! Taken from paper: https://www.sciencedirect.com/science/article/pii/S0022460X04001373
Cwx = 3. # ! Taken from paper: https://www.sciencedirect.com/science/article/pii/S0022460X04001373
Cuy = 10.
Cuz = Cuy
Cvy = 6.5
Cvz = Cvy
Cwy = 6.5
Cwz = 3.
    
# Defining Integral Length Scales:
L1 = 100 # m referanse length scale
z1 = 10  # m referanse height
zmin = 1 # m for Terrain Cat. 0 and I

xLu = np.zeros(num_nodes)
for n in list(range(num_nodes)):
    if(nodes[n,2]>zmin):
        xLu[n] = L1 * (nodes[n,2]/z1)**0.3
    else:
        xLu[n]= L1 * (zmin/z1)**0.3
yLu = 1/3 * xLu
zLu = 1/5 * xLu      
xLv = 1/4 * xLu
yLv = 1/4 * xLu
zLv = 1/12 * xLu
xLw = 1/12 * xLu
yLw = 1/18 * xLu
zLw = 1/18 * xLu

# Turbulence intensities
Iu = np.ones(num_nodes)*0.15
Iv = 0.85 * Iu
Iw = 0.55 * Iu

Ai = [Au, Av, Aw]
Cij = [Cux,Cvx,Cwx,Cuy,Cvy,Cwy,Cuz,Cvz,Cwz]
I = np.array([Iu,Iv,Iw])
I= np.moveaxis(I, -1, 0)
iLj = [xLu,yLu,zLu,xLv,yLv,zLv,xLw,yLw,zLw]

T = 3600*10 # sec. Time series duration. # best if multiple of block_duration. Good if 3600 sec
sample_freq = 4*0.1 # Hz # Good if 4 Hz
num_timepoints = T * sample_freq + 1

# =============================================================================
# Calling the wind_field_3D function
# =============================================================================
from wind_field_3D import wind_field_3D_func

wind_field_data = wind_field_3D_func(nodes, V, Ai, Cij, I, iLj, T, sample_freq, spectrum_type=2)

windspeed = wind_field_data["windspeed"]
windspeed_u = windspeed[0,:,:]-V[:,np.newaxis] # u component
timepoints = wind_field_data["timepoints"]
autospec_nondim = wind_field_data["autospec_nondim"]
cospec = wind_field_data["cospec"]
freq = wind_field_data["freq"]
delta_x, delta_y, delta_z = wind_field_data["delta_xyz"]

wind_field_data = None

# =============================================================================↨
#  Plotting MOVIE of wind across the g_nodes
# =============================================================================
#from matplotlib.animation import FuncAnimation
#import matplotlib
#matplotlib.use("Agg")
#from matplotlib.animation import FFMpegWriter
#fig, ax = plt.subplots()
#xdat1a, ydata1 = [], []
#xdata2, ydata2 = [], []
#xdata3, ydata3 = [], []
#ln1, = plt.plot([], [], animated=True, label='U')
#ln2, = plt.plot([], [], animated=True, label='v')
#ln3, = plt.plot([], [], animated=True, label='w')
#metadata = dict(title='Movie U', artist='Matplotlib', comment='Movie support!')
#writer = FFMpegWriter(fps=sample_freq, metadata=metadata)
#def init():
#    ax.set_xlim(0, num_nodes)
#    ax.set_ylim(-30, 100)
#    return ln1,
#    return ln2,
#    return ln3,
#def update(frame):    
#    xdata1 = list(range(num_nodes))
#    xdata2 = list(range(num_nodes))
#    xdata3 = list(range(num_nodes))
#    ydata1 = windspeed[0,:,frame] # U
#    ydata2 = windspeed[1,:,frame] # v
#    ydata3 = windspeed[2,:,frame] # w
#    ln1.set_data(xdata1, ydata1)
#    ln2.set_data(xdata2, ydata2)
#    ln3.set_data(xdata3, ydata3) 
#    return ln1,
#    return ln2,
#    return ln3,
#plt.title('Generated wind speed across the bridge g_nodes')
#plt.xlabel('Node number [-]')
#plt.ylabel('Wind speed [m/s]')
#plt.grid()
#plt.legend()
#ani = FuncAnimation(fig, update, frames=int(min(T*sample_freq,10*sample_freq)), init_func=init, blit=True)
#ani.save('Wind-Per-Node_02.mp4', writer=writer)
#plt.close()

# =============================================================================
# Plotting wind direction
# =============================================================================
plt.title('wind direction')
plt.plot(nodes[:,0],nodes[:,1])
#plt.arrow(g_nodes[num_nodes//2,0],g_nodes[num_nodes//2,1],500,0, head_width=300, head_length=300, head_starts_at_zero=False)
plt.arrow(0,0,500,0, head_width=300, head_length=300, head_starts_at_zero=False)
plt.annotate("",xy=(0,0),xytext=(0,0), arrowprops=dict(arrowstyle="->"))
plt.xlim([-chord,chord])
plt.ylim([-chord,chord])
plt.savefig('wind_direction')
plt.close()

# =============================================================================
# Plotting wind speed in 3 g_nodes
# =============================================================================
plt.plot(timepoints, windspeed[0,0,:], label='node 0')
plt.plot(timepoints, windspeed[0,1,:], label='node 1')
plt.plot(timepoints, windspeed[0,-1,:], label='last node')
plt.legend()
plt.xlabel('time [s]')
plt.ylabel('Wind speed [m/s]')
plt.xlim([0,100])
plt.grid()
plt.savefig('wind_speed_at_3_nodes')
plt.close()

# =============================================================================
# Plotting the means of wind velocities
# =============================================================================
plt.plot(list(range(num_nodes)), np.mean(windspeed[0,:,:], axis=1), 'rho', color='red', label='mean(U)', alpha=0.4)
plt.plot(V, 'rho', color='green', label='target', alpha=0.9)
plt.xlabel('time [s]')
plt.ylabel('Wind speed [m/s]')
plt.legend()
plt.grid()
plt.savefig('wind_speed_means')
plt.close()

# =============================================================================
# Plotting standard deviations of wind velocity
# =============================================================================
plt.scatter(list(range(num_nodes)), np.std(windspeed_u, axis=1), color='blue', label='std(u)')
plt.plot(Iu*V, color='blue', label='target std(u)')
plt.scatter(list(range(num_nodes)), np.std(windspeed[1,:,:], axis=1), color='orange', label='std(v)')
plt.plot(Iv*V, color='orange', label='target std(v)')
plt.scatter(list(range(num_nodes)), np.std(windspeed[2,:,:], axis=1), color='green', label='std(w)')
plt.plot(Iw*V, color='green', label='target std(w)')
plt.legend()
plt.gca().set_ylim(bottom=0)
plt.xlabel('Node number [-]')
plt.ylabel('std.(wind speed) [m/s]')
plt.grid()
plt.savefig('wind_speed_standard-deviations')
plt.close()

# =============================================================================
# Plotting auto-spectra - Non-dimensional
# =============================================================================
from scipy import signal
node_tested = 0 # node number to be tested
block_duration = min( 600 , T ) # s. Duration of each segment, to build an average in the Welch method
nperseg = len(windspeed_u[node_tested])/round(T/block_duration)
u_1_freq, u_1_ps = signal.welch(windspeed_u[node_tested] , sample_freq, nperseg=nperseg)
v_1_freq, v_1_ps = signal.welch(windspeed[1,node_tested] , sample_freq, nperseg=nperseg)
w_1_freq, w_1_ps = signal.welch(windspeed[2,node_tested] , sample_freq, nperseg=nperseg)
plt.title('Non-dimensional Auto-Spectrum') # Dimensional f
plt.plot(freq, autospec_nondim[0,node_tested], color='blue',  label='target u')
plt.plot(freq, autospec_nondim[1,node_tested], color='orange', label='target v')
plt.plot(freq, autospec_nondim[2,node_tested], color='green', label='target w')
plt.plot(u_1_freq, u_1_ps/((Iu[node_tested]*V[node_tested])**2)*u_1_freq, 'r--', color='blue', alpha=0.4, linewidth=0.5)
plt.plot(v_1_freq, v_1_ps/((Iv[node_tested]*V[node_tested])**2)*v_1_freq, 'r--', color='orange', alpha=0.4, linewidth=0.5)
plt.plot(w_1_freq, w_1_ps/((Iw[node_tested]*V[node_tested])**2)*w_1_freq, 'r--', color='green', alpha=0.4, linewidth=0.5)
plt.scatter(u_1_freq, u_1_ps/((Iu[node_tested]*V[node_tested])**2)*u_1_freq, color='blue', label='generated u', alpha=0.6, s =2)
plt.scatter(v_1_freq, v_1_ps/((Iv[node_tested]*V[node_tested])**2)*v_1_freq, color='orange', label='generated v', alpha=0.6, s =2)
plt.scatter(w_1_freq, w_1_ps/((Iw[node_tested]*V[node_tested])**2)*w_1_freq, color='green', label='generated w', alpha=0.6, s =2)
plt.xscale('log')
plt.xlabel('freq [Hz]')
plt.xlim([1/T/1.2 , sample_freq/2*1.2]) # 1.2 zoom-out ratio
plt.legend()
plt.grid(which='both')
plt.savefig('Non-dim-auto-spectrum_standard_VS_measurements')
plt.close()

# =============================================================================
# Plotting auto-spectra - Dimensional
# =============================================================================
node_tested = 0 # node number to be tested
block_duration = min( 600 , T ) # s. Duration of each segment, to build an average in the Welch method
nperseg = len(windspeed_u[node_tested])/round(T/block_duration)
u_1_freq, u_1_ps = signal.welch(windspeed_u[node_tested] , sample_freq, nperseg=nperseg)
v_1_freq, v_1_ps = signal.welch(windspeed[1,node_tested] , sample_freq, nperseg=nperseg)
w_1_freq, w_1_ps = signal.welch(windspeed[2,node_tested] , sample_freq, nperseg=nperseg)
plt.title('Auto-Spectrum S(f)') # Dimensional f
plt.plot(freq, autospec_nondim[0,node_tested]*((Iu[node_tested]*V[node_tested])**2)/freq, color='blue',  label='target u')
plt.plot(freq, autospec_nondim[1,node_tested]*((Iv[node_tested]*V[node_tested])**2)/freq, color='orange', label='target v')
plt.plot(freq, autospec_nondim[2,node_tested]*((Iw[node_tested]*V[node_tested])**2)/freq, color='green', label='target w')
plt.plot(u_1_freq, u_1_ps, 'r--', color='blue', alpha=0.4, linewidth=0.5)
plt.plot(v_1_freq, v_1_ps, 'r--', color='orange', alpha=0.4, linewidth=0.5)
plt.plot(w_1_freq, w_1_ps, 'r--', color='green', alpha=0.4, linewidth=0.5)
plt.scatter(u_1_freq, u_1_ps, color='blue', label='generated u', alpha=0.6, s =2)
plt.scatter(v_1_freq, v_1_ps, color='orange', label='generated v', alpha=0.6, s =2)
plt.scatter(w_1_freq, w_1_ps, color='green', label='generated w', alpha=0.6, s =2)
plt.xscale('log')
plt.xlabel('freq [Hz]')
plt.xlim([1/T/1.2 , sample_freq/2*1.2]) # 1.2 zoom-out ratio
plt.legend()
plt.grid(which='both')
plt.savefig('Dim-auto-spectrum_standard_VS_measurements')
plt.close()

# =============================================================================
# Plotting coherence, function of freq and spatial separation
# =============================================================================
from coherence import coherence

n_coh = 6 # number of g_nodes to assess coherence. n_coh <= num_nodes

fig = plt.figure(figsize=(6*n_coh,6*n_coh))
nperseg = 256 # Welch's method. Length of each segment.
counter=0
for node_1 in range(n_coh):
    for node_2 in range(n_coh):
        counter+=1

        if node_2 > node_1:
            ax = fig.add_subplot(n_coh,n_coh,counter)
            coh_freq_u = coherence(windspeed_u[node_1], windspeed_u[node_2], fs=sample_freq, nperseg=nperseg)['freq']
            coh_u = coherence(windspeed_u[node_1], windspeed_u[node_2], fs=sample_freq, nperseg=nperseg)['cocoh']
            ax.set_title('Node '+str(node_1)+' and '+str(node_2) )
            ax.plot(freq, np.e**( -np.sqrt((Cux*delta_x[node_1,node_2])**2 +
                                            (Cuy*delta_y[node_1,node_2])**2 + 
                                            (Cuz*delta_z[node_1,node_2])**2 ) * 
                                            freq/((V[node_1]+V[node_2])/2) ), color='blue', label='target u', alpha=0.75)
            plt.scatter(coh_freq_u, coh_u, alpha=0.6, s=4, color='blue', label='generated u')
            
            coh_freq_v = coherence(windspeed[1,node_1], windspeed[1,node_2], fs=sample_freq, nperseg=nperseg)['freq']
            coh_v = coherence(windspeed[1,node_1], windspeed[1,node_2], fs=sample_freq, nperseg=nperseg)['cocoh']
            plt.plot(freq, np.e**( -np.sqrt((Cvx*delta_x[node_1,node_2])**2 +
                                            (Cvy*delta_y[node_1,node_2])**2 + 
                                            (Cvz*delta_z[node_1,node_2])**2 ) * 
                                            freq/((V[node_1]+V[node_2])/2) ), color='orange', label='target v', alpha=0.75)
            plt.scatter(coh_freq_v, coh_v, alpha=0.6, s=4, color='orange', label='generated v')
            
            coh_freq_w = coherence(windspeed[2,node_1], windspeed[2,node_2], fs=sample_freq, nperseg=nperseg)['freq']
            coh_w = coherence(windspeed[2,node_1], windspeed[2,node_2], fs=sample_freq, nperseg=nperseg)['cocoh']
            plt.plot(freq, np.e**( -np.sqrt((Cwx*delta_x[node_1,node_2])**2 +
                                            (Cwy*delta_y[node_1,node_2])**2 + 
                                            (Cwz*delta_z[node_1,node_2])**2 ) * 
                                            freq/((V[node_1]+V[node_2])/2) ), color='green', label='target w', alpha=0.75)
            plt.scatter(coh_freq_w, coh_w, alpha=0.6, s=4, color='green', label='generated w')
            plt.xlabel('freq [Hz]')
            plt.ylabel('Coherence')
            plt.grid()
            plt.legend()
plt.tight_layout()
plt.savefig('Co-coherence')
plt.close()

# =============================================================================
# Plotting correlation coefficients between g_nodes (function of spatial separation in x,y and z)
# =============================================================================
corrcoef = np.zeros((num_nodes,num_nodes))
corrcoef_u_target = np.zeros((num_nodes,num_nodes))
corrcoef_v_target = np.zeros((num_nodes,num_nodes))
corrcoef_w_target = np.zeros((num_nodes,num_nodes))
for n in range(len(nodes)):
    for m in range(len(nodes)):
        corrcoef_u_target[n,m] = np.exp( -np.sqrt( (delta_x[n,m]/((xLu[n]+xLu[m])/2))**2 + (delta_y[n,m]/((yLu[n]+yLu[m])/2))**2 + (delta_z[n,m]/((zLu[n]+zLu[m])/2))**2  ))
        corrcoef_v_target[n,m] = np.exp( -np.sqrt( (delta_x[n,m]/((xLv[n]+xLv[m])/2))**2 + (delta_y[n,m]/((yLv[n]+yLv[m])/2))**2 + (delta_z[n,m]/((zLv[n]+zLv[m])/2))**2  ))
        corrcoef_w_target[n,m] = np.exp( -np.sqrt( (delta_x[n,m]/((xLw[n]+xLw[m])/2))**2 + (delta_y[n,m]/((yLw[n]+yLw[m])/2))**2 + (delta_z[n,m]/((zLw[n]+zLw[m])/2))**2  ))
corrcoef_u = np.corrcoef(windspeed_u)
corrcoef_v = np.corrcoef(windspeed[1])
corrcoef_w = np.corrcoef(windspeed[2])
corrcoef = np.array([corrcoef_u, corrcoef_v, corrcoef_w])
corrcoef_target = np.array([corrcoef_u_target, corrcoef_v_target, corrcoef_w_target])
from mpl_toolkits.mplot3d import Axes3D
del Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
X, Y = np.meshgrid(list(range(len(nodes_x))),list(range(len(nodes_y))))
colors=[[],[],[]]
for i in range(3):
    colors[i]=cm.rainbow(abs(corrcoef[i]-corrcoef_target[i]))   ##### corrcoef[i]-corrcoef_target[i]
fig = plt.figure(figsize=(16, 6))
plt.suptitle('Top: Correlation coefficients \n Bottom: Corr. coeff. difference between expected and generated')
for i in range(3):
    ax = fig.add_subplot(2,3,i+1, projection='3d')
    ax = fig.gca(projection='3d')
    ax.set_title(str(['u','v','w'][i]), fontweight='bold')
    surf1 = ax.plot_surface(X,Y, corrcoef[i], linewidth=0.2, antialiased=True, color='orange', alpha=0.9, label='generated')
    ax.plot_surface(X,Y, corrcoef_target[i], linewidth=0.2, antialiased=True, color='green', alpha=0.5, label='expected')
    ax.set_xlabel('Node Number')
    ax.set_ylabel('Node Number')
    ax.set_zlabel('corr. coeff.')
    ax.view_init(elev=20, azim=240)
    ax.text(num_nodes, 0, 0.15, "generated", color='orange')
    ax.text(num_nodes, 0, 0, "expected", color='green')
for i in range(3): 
    ax = fig.add_subplot(2,3,i+4)
    ax = fig.gca()  
    img = ax.imshow(colors[i], cmap=cm.rainbow)
    plt.gca().invert_yaxis()
    cbar = plt.colorbar(img)
    ax.set_xlabel('Node Number')
    ax.set_ylabel('Node Number')
    cbar.set_label('$\Delta$ corr. coeff.')    
plt.savefig('Correlation_coefficients')
plt.close()

# =============================================================================
# 
# =============================================================================
