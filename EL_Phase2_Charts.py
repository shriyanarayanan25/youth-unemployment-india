"""
EL Phase II: draws the charts. Run AFTER (or independently of) EL_Phase2_Analysis_Script.py.
Run:  python EL_Phase2_Charts.py      (keep EL_Phase2_Master_Table.csv in the same folder)
Needs: pandas numpy scikit-learn matplotlib
Saves PNG files into a folder called charts/
"""
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

os.makedirs("charts", exist_ok=True)
df = pd.read_csv("EL_Phase2_Master_Table.csv")
df["log_nsdp"] = np.log(df["pc_nsdp_real"])
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

# 1. Trend over time: average youth UR per year with the middle-50% band
g = df.groupby("year")["youth_ur_total"]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.fill_between(g.mean().index, g.quantile(.25), g.quantile(.75), color=BLUE, alpha=.15, label="Middle 50% of states")
ax.plot(g.mean().index, g.mean(), marker="o", color=BLUE, label="Average of 30 states")
ax.set_ylabel("Youth (15-29) unemployment rate, %"); ax.set_title("Youth unemployment by year", loc="left", weight="bold")
ax.legend(frameon=False); fig.tight_layout(); fig.savefig("charts/fig1_trend.png", dpi=200); plt.close()

# 2. State averages (7-year mean), small states shown lighter
m = df.groupby("state").agg(ur=("youth_ur_total", "mean"), small=("small_state", "max")).sort_values("ur")
fig, ax = plt.subplots(figsize=(7, 8))
ax.barh(m.index, m.ur, color=[("#9ec5f4" if s else BLUE) for s in m.small])
ax.set_xlabel("Average youth unemployment rate, 2017-18 to 2023-24 (%)")
ax.set_title("Youth unemployment by state (light bars = small states, noisier)", loc="left", weight="bold")
fig.tight_layout(); fig.savefig("charts/fig2_state_means.png", dpi=200); plt.close()

# 3. Correlation heatmap
cols = ["youth_ur_total", "log_nsdp", "higher_ed_ger", "tele_density", "female_lfpr_youth", "share_industry", "share_services"]
c = df[cols].corr()
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right"); ax.set_yticks(range(len(cols)), cols)
for i in range(len(cols)):
    for j in range(len(cols)): ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax.set_title("Correlations", loc="left", weight="bold"); fig.colorbar(im, shrink=.8)
fig.tight_layout(); fig.savefig("charts/fig3_correlations.png", dpi=200); plt.close()

# 4-6. Clustering (same settings as the analysis script)
FEATURES = ["youth_ur_total", "log_nsdp", "higher_ed_ger", "female_lfpr_youth", "share_agri", "share_industry"]
state = df.groupby("state")[FEATURES].mean()
Z = StandardScaler().fit_transform(state)
K = 4
labels = KMeans(K, n_init=200, random_state=42).fit_predict(Z)
state["cluster"] = labels
# name clusters by their average youth UR rank so colours are stable
order = state.groupby("cluster")["youth_ur_total"].mean().sort_values().index.tolist()
state["cl"] = state["cluster"].map({c_: i for i, c_ in enumerate(order)})
COL = [BLUE, ORANGE, AQUA, YELLOW]; MK = ["o", "s", "^", "D"]

pca = PCA(2).fit(Z); P = pca.transform(Z); ev = pca.explained_variance_ratio_
fig, ax = plt.subplots(figsize=(10, 7.5))
for k in range(K):
    i = (state.cl == k).values
    ax.scatter(P[i, 0], P[i, 1], c=COL[k], marker=MK[k], s=70, label=f"Cluster {k+1} (avg UR {state.youth_ur_total[i].mean():.1f}%)")
for (x, y), name in zip(P, state.index): ax.annotate(name, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=7)
ax.set_xlabel(f"PC1 ({ev[0]:.0%} of variation)"); ax.set_ylabel(f"PC2 ({ev[1]:.0%})")
ax.set_title("State clusters (K-means, 7-year averages)", loc="left", weight="bold"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("charts/fig5_state_clusters.png", dpi=200); plt.close()

prof = state.groupby("cl")[FEATURES].mean()
zp = (prof - state[FEATURES].mean()) / state[FEATURES].std()
fig, ax = plt.subplots(figsize=(8, 4))
ax.imshow(zp, cmap="RdBu_r", vmin=-2, vmax=2)
ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=35, ha="right"); ax.set_yticks(range(K), [f"Cluster {i+1}" for i in range(K)])
for i in range(K):
    for j in range(len(FEATURES)): ax.text(j, i, f"{prof.iloc[i, j]:.1f}", ha="center", va="center", fontsize=8)
ax.set_title("Cluster averages (colour = above/below all-state average)", loc="left", weight="bold")
fig.tight_layout(); fig.savefig("charts/fig6_cluster_profiles.png", dpi=200); plt.close()

ks = range(2, 7)
inertia, sil = [], []
for k in ks:
    km = KMeans(k, n_init=100, random_state=42).fit(Z); inertia.append(km.inertia_); sil.append(silhouette_score(Z, km.labels_))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
a1.plot(ks, inertia, marker="o", color=BLUE); a1.set_title("Elbow (inertia)"); a1.set_xlabel("k")
a2.plot(ks, sil, marker="o", color=ORANGE); a2.set_title("Silhouette (higher = clearer)"); a2.set_xlabel("k")
fig.tight_layout(); fig.savefig("charts/fig7_choosing_k.png", dpi=200); plt.close()

# 4. Regression effects (needs statsmodels): effect of a 1-SD rise in each predictor, with 95% interval
try:
    import statsmodels.formula.api as smf
    X = ["log_nsdp", "higher_ed_ger", "tele_density", "female_lfpr_youth", "share_industry", "share_services"]
    d = df.dropna(subset=X + ["youth_ur_total"])
    r = smf.ols("youth_ur_total ~ " + " + ".join(X) + " + C(state) + C(year)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d.state)[0]})
    sd = d[X].std(); b = r.params[X] * sd; ci = r.conf_int().loc[X].mul(sd, axis=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(b, range(len(X)), xerr=[b - ci[0], ci[1] - b], fmt="o", color=BLUE, capsize=3)
    ax.axvline(0, color="grey", lw=.8); ax.set_yticks(range(len(X)), X)
    ax.set_xlabel("Change in youth UR (percentage points) for a 1-SD rise in the variable")
    ax.set_title("Two-way fixed effects: effect sizes with 95% intervals", loc="left", weight="bold")
    fig.tight_layout(); fig.savefig("charts/fig4_regression_effects.png", dpi=200); plt.close()
except ImportError:
    print("statsmodels not installed: skipped fig4 (pip install statsmodels)")
print("Done. Charts are in the charts/ folder.")
