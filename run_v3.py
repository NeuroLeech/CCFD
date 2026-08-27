"""Minimal: two regions, timecourses made by hand, run, save an mp4."""
import numpy as np
from mesh_cache import load_cortex
from input2 import RegionDrive
from swe_rot import RotSWE, sponge_profile
import numpy as np

cortex = load_cortex("fsaverage5")          # this is `cortex`

#REGIONS = [9,1,24, 150,30,65,132]                           # V1, 10r   (cortex.names[1] == 'L_V1_ROI')
#REGIONS = [9,1,24, 150,30,65,132]                           # V1, 10r   (cortex.names[1] == 'L_V1_ROI')
REGIONS = [9,1,24]# 150,30,65,132]                           # V1, 10r   (cortex.names[1] == 'L_V1_ROI')
#REGIONS = [9,1,24, 150 30,30,65]
#REGIONS = [9,65,132, 24,30,1,150]                           # V1, 10r   (cortex.names[1] == 'L_V1_ROI')
NSTEPS = 6000

def numpy_ema(prices, span):
    alpha = 2 / (span + 1)
    ema = np.zeros_like(prices, dtype=float)
    
    # Seed initial value with first data point
    ema[0] = prices[0]
    
    for t in range(1, len(prices)):
        ema[t] = alpha * prices[t] + (1 - alpha) * ema[t - 1]
        
    return ema

CFL, C, G, H = 0.2, 1.0, 1.0, 1.0
dt = CFL*cortex.d.min()/C                   # 0.347 time units; wave crossing ~200

# --- your timecourses: (NSTEPS, 2), one column per region -------------------
t = np.arange(NSTEPS)*dt

N1 = 10        # Number of boxcars
N2 = NSTEPS        # Total array size
min_len = 100     # Minimum boxcar length
offset=25
gap=300
#max_len = 250    # Maximum boxcar length


# --- GENERATE BOXCARS ---
signal = np.zeros(N2)
for i in range(N1):
    if np.random.randn(1)<0:
        signal[gap*i +offset:gap*i +offset+min_len]=1
    else:
        signal[gap*i +offset:gap*i +offset+min_len]=-1


signal=numpy_ema(signal, span=50)

#A = np.c_[ np.cos(2*np.pi*t/300),np.cos(2*np.pi*t/300),
#            np.cos(2*np.pi*t/300),np.cos(2*np.pi*t/300)*-1,np.cos(2*np.pi*t/300)*-1,np.cos(2*np.pi*t/300)*-1,np.cos(2*np.pi*t/300)*-1]

A = np.c_[ np.cos(2*np.pi*t/300),np.cos(2*np.pi*t/300),np.cos(2*np.pi*t/300)]

#A = np.c_[ signal*-1,signal*-1, signal*-1,signal*1,signal*1,signal*1,signal*1]

# ---------------------------------------------------------------------------
print(t)
print(A)
drive = RegionDrive(cortex, REGIONS, A, amp=2e-4, nsteps=NSTEPS)
drive.describe()

# s = RotSWE(cortex.m, C/52.4, l=cortex.l, d=cortex.d, A=cortex.A,
#            E=cortex.edges, bnd_edge=cortex.bnd)      # 52.4 = Ld, the frozen regime
s = RotSWE(cortex.m, C/52.4, l=cortex.l, d=cortex.d, A=cortex.A,
           E=cortex.edges, bnd_edge=cortex.bnd)      
#s.set_sponge(sponge_profile(cortex.V, cortex.edges, cortex.bnd, 7.25, 0.27) + 0.001)
#s.set_sponge(np.zeros(cortex.nV))
#s.sig_e[:] = 0.001          # drag on velocity only

s.astype(np.float32)

Aser = drive.Aser.astype(np.float32)
P = drive.P.astype(np.float32)
h = np.zeros(cortex.nV, np.float32)
ue = np.zeros(s.nE, np.float32)
frames = []
for n in range(NSTEPS):
    h += Aser[n] @ P                                  # inject
    ue, h = s.step(ue, h, np.float32(dt), np.float32(G), np.float32(H))
    if n % 25 == 0:
        frames.append(h.copy())
frames = np.array(frames)
print(f"  {len(frames)} frames, peak {100*np.abs(frames).max()/H:.2f}% of depth")

np.save('results/frames.npy',frames)
# --- video ------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.animation as animation
from render_regimes import _proj

proj = _proj(cortex.V, cortex.F)
vl = np.percentile(np.abs(frames), 96)
fig, axes = plt.subplots(1, len(proj), figsize=(4*len(proj), 4))
scs = []
for ax, (xy, vis, nm) in zip(axes, proj):
    scs.append(ax.scatter(xy[vis, 0], xy[vis, 1], c=np.zeros(vis.sum()),
                          s=3, linewidths=0, cmap="RdBu_r", vmin=-vl, vmax=vl))
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(nm)

def upd(i):
    for sc, (_, vis, _) in zip(scs, proj):
        sc.set_array(frames[i][vis])
    return []
vid_output="results/videos/run2_v4.mp4"
animation.FuncAnimation(fig, upd, frames=len(frames)).save(
    vid_output, writer=animation.FFMpegWriter(fps=16))
print("  wrote " + vid_output)
