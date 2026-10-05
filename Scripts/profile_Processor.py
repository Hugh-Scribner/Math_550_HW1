import re
import glob
import numpy as np
import pstats
import matplotlib.pyplot as plt

def get_cumtime(stats, funcname, filename_contains=None):
    """Cumulative time of the first function matching funcname (summed over matches)."""
    total = 0.0
    for (fname, lineno, name), (cc, nc, tt, ct, callers) in stats.stats.items():
        if name == funcname and (filename_contains is None or filename_contains in fname):
            total += ct
    return total

# Load files; assumes names like stokes_50.prof, stokes_100.prof, ...
files = sorted(glob.glob("stokes_*.prof"),
               key=lambda f: int(re.search(r"stokes_(\d+)\.prof", f).group(1)))
meshes = np.array([int(re.search(r"stokes_(\d+)\.prof", f).group(1)) for f in files])

# Pick out the stages you care about
stages = {
    "Total (main)":        lambda s: get_cumtime(s, "main"),
    "Assembly":            lambda s: get_cumtime(s, "sysAssembly"),
    "Solve (spsolve)":     lambda s: get_cumtime(s, "spsolve"),
    "buildLaplacian":      lambda s: get_cumtime(s, "buildLaplacian"),
    "buildDeriv":          lambda s: get_cumtime(s, "buildDeriv"),
}

times = {name: [] for name in stages}
for f in files:
    s = pstats.Stats(f)
    for name, fn in stages.items():
        times[name].append(fn(s))

# Plot 1: time per stage vs mesh size, log-log, with reference slopes
fig, ax = plt.subplots(figsize=(7, 5))
for name, t in times.items():
    ax.loglog(meshes, t, "o-", label=name)

# Reference lines anchored at the first data point of the solve
t0 = times["Solve (spsolve)"][0]
ax.loglog(meshes, t0 * (meshes / meshes[0])**2, "k--", alpha=0.5,
          label=r"$\mathcal{O}(N^2)$ Reference")
ax.loglog(meshes, t0 * (meshes / meshes[0])**3, "k:", alpha=0.5,
          label=r"$\mathcal{O}(N^3)$ Reference")

ax.set_xlabel("Number of Nodes")
ax.set_ylabel("Time (s)")
ax.set_title("Runtime by stage")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
plt.savefig("Images\\Runtime_by_stage.png", bbox_inches='tight')

# Plot 2: fraction of time in assembly vs solve (stacked bars)
fig2, ax2 = plt.subplots(figsize=(7, 5))
asm = np.array(times["Assembly"])
sol = np.array(times["Solve (spsolve)"])
frac_asm = asm / (asm + sol)
x = np.arange(len(meshes))
ax2.bar(x, frac_asm, label="Assembly")
ax2.bar(x, 1 - frac_asm, bottom=frac_asm, label="Solve")
ax2.set_xticks(x)
ax2.set_xticklabels(meshes)
ax2.set_xlabel("Number of nodes")
ax2.set_ylabel("Percentage of Total Solve Time")
ax2.legend()
fig2.tight_layout()
plt.savefig("Images\\runtime_barplot.png", bbox_inches='tight')
