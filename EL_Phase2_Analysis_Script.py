"""
EL Phase II: panel regression and clustering on the master table.
Run:  python EL_Phase2_Analysis_Script.py        (keep EL_Phase2_Master_Table.csv in the same folder)
Needs: pandas numpy scipy scikit-learn statsmodels   (pip install statsmodels)
"""
import numpy as np
import pandas as pd

df = pd.read_csv("EL_Phase2_Master_Table.csv")

# ------------------------------------------------------------------
# 0. PREPARE
# ------------------------------------------------------------------
df["log_nsdp"] = np.log(df["pc_nsdp_real"])        # income in logs: a % change matters more than a rupee change

# Predictors. Agriculture share is left out on purpose: agri + industry + services = 100%,
# so including all three would make the model impossible to estimate.
# gender_gap is NOT used by default: it is female UR minus male UR, which is built from the same
# numbers as the outcome. Replace it with female youth labour-force participation once you have it.
XVARS = ["log_nsdp", "higher_ed_ger", "tele_density", "share_industry", "share_services"]
Y = "youth_ur_total"

print("\n=== DESCRIPTIVES ===")
print(df[[Y] + XVARS].describe().round(2))
print("\nAverage youth UR by year:\n", df.groupby("year")[Y].mean().round(1))
print("\nAverage youth UR by state (highest first):\n", df.groupby("state")[Y].mean().sort_values(ascending=False).round(1))
print("\nCorrelations:\n", df[[Y] + XVARS].corr().round(2))

# ------------------------------------------------------------------
# 1. PANEL REGRESSION
# ------------------------------------------------------------------
# Idea: the same 30 states are observed for 7 years. Fixed effects give every state its own baseline
# (removes anything permanent about a state, e.g. geography or culture) and every year its own baseline
# (removes shocks hitting all states, e.g. COVID). What is left is: when a state changes over time,
# does unemployment move with its other variables?
#   UR_it = a + b'X_it + state_i + year_t + error_it
try:
    import statsmodels.formula.api as smf
    d = df.dropna(subset=XVARS + [Y]).copy()          # Sikkim has no tele_density, so it drops out here
    rhs = " + ".join(XVARS)
    clust = {"groups": pd.factorize(d["state"])[0]}   # standard errors clustered by state

    pooled = smf.ols(f"{Y} ~ {rhs}", data=d).fit(cov_type="cluster", cov_kwds=clust)
    state_fe = smf.ols(f"{Y} ~ {rhs} + C(state)", data=d).fit(cov_type="cluster", cov_kwds=clust)
    twfe = smf.ols(f"{Y} ~ {rhs} + C(state) + C(year)", data=d).fit(cov_type="cluster", cov_kwds=clust)  # MAIN MODEL

    def show(res, name):
        keep = [v for v in res.params.index if v in XVARS]
        t = pd.DataFrame({"coef": res.params[keep], "std err": res.bse[keep], "p-value": res.pvalues[keep]}).round(3)
        print(f"\n=== {name}  (n={int(res.nobs)}, R2={res.rsquared:.3f}) ===\n", t)

    show(pooled, "Pooled OLS")
    show(state_fe, "State fixed effects")
    show(twfe, "Two-way fixed effects (main)")

    # Fixed effects vs random effects. Mundlak test: add each predictor's state average to the model and
    # test whether those averages matter. If they do (small p), fixed effects is the safer choice.
    for v in XVARS:
        d[v + "_mean"] = d.groupby("state")[v].transform("mean")
    mund_terms = [v + "_mean" for v in XVARS]
    mund = smf.ols(f"{Y} ~ {rhs} + {' + '.join(mund_terms)} + C(year)", data=d).fit(cov_type="cluster", cov_kwds=clust)
    print("\nMundlak test (joint):", mund.wald_test(", ".join(f"{t} = 0" for t in mund_terms), scalar=True))

    # Multicollinearity: VIF above ~5-10 means predictors overlap too much.
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    Xv = d[XVARS].assign(const=1.0)
    print("\nVIF:", {v: round(variance_inflation_factor(Xv.values, i), 2) for i, v in enumerate(XVARS)})

    # Robustness: rerun the main model without the 9 small states.
    d2 = d[d["small_state"] == 0]
    r2 = smf.ols(f"{Y} ~ {rhs} + C(state) + C(year)", data=d2).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d2["state"])[0]})
    show(r2, "Two-way FE without small states")
except ImportError:
    print("\n[statsmodels not installed: run  pip install statsmodels  to get the regression part]")

# ------------------------------------------------------------------
# 2. CLUSTERING
# ------------------------------------------------------------------
# Idea: describe each state by a few averages, put them on the same scale, and let K-means group similar states.
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["youth_ur_total", "log_nsdp", "higher_ed_ger", "share_agri", "share_industry"]   # add female participation once available
state = df.groupby("state")[FEATURES].mean()                    # one row per state: its 7-year average
Z = StandardScaler().fit_transform(state)                       # same scale: mean 0, std 1

print("\n=== CHOOSING k ===\n k   inertia  silhouette")
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=200, random_state=42).fit(Z)
    print(f" {k}  {km.inertia_:8.1f}  {silhouette_score(Z, km.labels_):.3f}")
# Look for an elbow in inertia and a high silhouette (above ~0.5 = clear clusters, 0.25-0.5 = weak structure).

K = 3
labels = KMeans(n_clusters=K, n_init=200, random_state=42).fit_predict(Z)
state["cluster"] = labels
print("\n=== CLUSTER AVERAGES ===\n", state.groupby("cluster")[FEATURES].mean().round(1).assign(n=state.groupby("cluster").size()))
for c in range(K):
    print(f"Cluster {c}:", ", ".join(state.index[state.cluster == c]))

# Check the groups are not a fluke: compare with a different method, and with resampling.
ward = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(Z)
print("\nAgreement with Ward hierarchical clustering (ARI, 1 = identical):", round(adjusted_rand_score(labels, ward), 2))
rng = np.random.default_rng(0)
ari = []
for _ in range(300):
    idx = rng.choice(len(Z), size=int(0.8 * len(Z)), replace=False)
    sub = KMeans(n_clusters=K, n_init=20, random_state=1).fit_predict(Z[idx])
    ari.append(adjusted_rand_score(labels[idx], sub))
print("Stability under 80% resampling (average ARI):", round(float(np.mean(ari)), 2))

# 2-D picture of the states (only for plotting; clustering used all features)
pca = PCA(2).fit(Z)
state[["PC1", "PC2"]] = pca.transform(Z)
print("\nShare of variation shown by the 2-D picture:", pca.explained_variance_ratio_.round(2))
state.to_csv("state_clusters.csv")
print("\nSaved state_clusters.csv")
