import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.stats import spearmanr, mannwhitneyu

LEXACOM_FILE = "LExACoM_MRT_Accepted.txt"
NASA_SNAPSHOT = "nasa_ps_snapshot.csv"
OUTPUT_CSV = "matched_ccr_systems.csv"
OUTPUT_PLOT = "CCR_gradient_plot.png"
SOLAR_CO = 0.549

def log(msg):
    print("[INFO] " + str(msg))

def normalize(s):
    return s.str.lower().str.replace(r"[\s\-_]", "", regex=True).str.strip()

def hkey(name):
    return name.lower().replace(" ","").replace("-","").replace("_","")

STELLAR_CO = {
    "hd189733":  (0.490, 0.060),
    "hd209458":  (0.540, 0.050),
    "hd149026":  (0.580, 0.060),
    "wasp18":    (0.530, 0.070),
    "wasp77a":   (0.490, 0.060),
    "wasp121":   (0.520, 0.060),
    "hd80606":   (0.540, 0.070),
    "wasp127":   (0.510, 0.060),
    "wasp80":    (0.500, 0.080),
    "gj3470":    (0.490, 0.090),
    "wasp107":   (0.540, 0.060),
    "wasp76":    (0.560, 0.060),
    "wasp69":    (0.510, 0.070),
    "wasp166":   (0.530, 0.060),
    "tres4":     (0.520, 0.070),
    "wasp15":    (0.540, 0.060),
    "hatp14":    (0.530, 0.060),
    "mascara1":  (0.520, 0.070),
    "kelt20":    (0.550, 0.060),
    "tauboo":    (0.840, 0.060),
    "wasp19":    (0.510, 0.070),
    "wasp33":    (0.550, 0.070),
    "wasp43":    (0.490, 0.080),
    "wasp94a":   (0.530, 0.060),
    "wasp94":    (0.530, 0.060),
    "wasp189":   (0.540, 0.060),
    "v1298tau":  (0.530, 0.080),
    "kelt7":     (0.560, 0.060),
    "hr8799":    (0.540, 0.060),
    "betapic":   (0.550, 0.070),
    "aflep":     (0.530, 0.060),
    "pds70":     (0.510, 0.080),
    "roxs42b":   (0.520, 0.080),
    "roxs12":    (0.510, 0.080),
    "gqlup":     (0.510, 0.080),
    "dhtau":     (0.520, 0.080),
    "wasp178":   (0.560, 0.060),
    "toi5205":   (0.490, 0.080),
    "hip65":     (0.520, 0.070),
    "kapand":    (0.520, 0.080),
    "gsc6214210":(0.510, 0.090),
    "2m0122":    (0.520, 0.090),
    "abpic":     (0.520, 0.090),
    "51eri":     (0.540, 0.060),
    "wd0806":    (0.520, 0.090),
}

def get_stellar_co(hostname):
    k = hkey(hostname)
    if k in STELLAR_CO:
        return STELLAR_CO[k]
    for key, val in STELLAR_CO.items():
        if k.startswith(key) or key.startswith(k[:min(6,len(k))]):
            return val
    return None

log("Loading LExACoM...")
lex = Table.read(LEXACOM_FILE, format="ascii.mrt").to_pandas()
lex = lex.rename(columns={"Planet": "pl_name", "Geometry": "obs_geometry"})
lex = lex[["pl_name","C/O","e_C/O","E_C/O","obs_geometry"]].dropna(subset=["C/O"])
lex = lex[lex["C/O"] > 0].copy()
lex["pl_name_norm"] = normalize(lex["pl_name"])
lex["hostname"] = lex["pl_name"].apply(lambda x: " ".join(x.split()[:-1]))
log("LExACoM rows: " + str(len(lex)))

log("Loading NASA snapshot...")
ps = pd.read_csv(NASA_SNAPSHOT)
ps["pl_name_norm"] = normalize(ps["pl_name"])
ps_orb = ps.dropna(subset=["pl_orbsmax"]).groupby("pl_name_norm")["pl_orbsmax"].first()
log("NASA planets: " + str(len(ps)))

log("Matching stellar C/O from literature...")
rows = []
no_match = []
for _, row in lex.iterrows():
    co_data = get_stellar_co(row["hostname"])
    if co_data is None:
        no_match.append(row["hostname"])
        continue
    co_star, sigma_star = co_data
    ratio = row["C/O"] / co_star
    if ratio <= 0:
        continue
    ccr = abs(np.log10(ratio))
    sigma_p = (abs(row["e_C/O"]) + abs(row["E_C/O"])) / 2
    sigma_ccr = (1.0/np.log(10)) * np.sqrt(
        (sigma_p / row["C/O"])**2 +
        (sigma_star / co_star)**2
    )
    pnorm = row["pl_name_norm"]
    orbsmax = ps_orb.get(pnorm, np.nan)
    rows.append({"pl_name":row["pl_name"],"hostname":row["hostname"],"obs_geometry":row["obs_geometry"],"C/O":row["C/O"],"e_C/O":row["e_C/O"],"E_C/O":row["E_C/O"],"C_O_star":co_star,"sigma_star":sigma_star,"CCR":ccr,"sigma_CCR":sigma_ccr,"pl_orbsmax":orbsmax})

if no_match:
    log("No stellar C/O for: " + str(list(set(no_match))))

an = pd.DataFrame(rows)
an = an.dropna(subset=["pl_orbsmax","CCR"]).copy()
an = an[an["pl_orbsmax"] > 0].copy()
log("Matched rows: " + str(len(an)))
log("Unique planets: " + str(an["pl_name"].nunique()))

if len(an) < 3:
    raise SystemExit("Not enough matches")

rho, pval = spearmanr(np.log10(an["pl_orbsmax"]), an["CCR"])
log("Spearman rho=" + str(round(rho,4)) + "  p=" + str(round(pval,6)))

log("CCR by geometry:")
for g, grp in an.groupby("obs_geometry"):
    log("  " + str(g) + "  N=" + str(len(grp)) + "  mean=" + str(round(grp["CCR"].mean(),4)))

geoms = sorted(an["obs_geometry"].unique())
if len(geoms) >= 2:
    g1 = an[an["obs_geometry"]==geoms[0]]["CCR"].values
    g2 = an[an["obs_geometry"]==geoms[1]]["CCR"].values
    if len(g1)>1 and len(g2)>1:
        u_stat,u_p=mannwhitneyu(g1,g2,alternative="two-sided")
        log("Mann-Whitney p="+str(round(u_p,4)))

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,6))
colors={"Transit":"steelblue","Eclipse":"tomato","Direct":"darkgreen","transit":"steelblue","eclipse":"tomato","direct":"darkgreen"}
for g,grp in an.groupby("obs_geometry"):
    c=colors.get(g,"gray")
    ax1.errorbar(grp["pl_orbsmax"],grp["CCR"],yerr=grp["sigma_CCR"],fmt="o",color=c,label=g,capsize=3,alpha=0.8)
ss_a=[0.387,0.723,1.0,1.524,5.203,9.537,19.191,30.069]
ss_ccr=[0.541,0.524,0.514,0.494,0.009,0.071,0.101,0.131]
ax1.scatter(ss_a,ss_ccr,marker="*",s=200,color="gold",zorder=10,label="Solar system",edgecolors="black",linewidths=0.8)
ax1.axhline(0.0,color="black",linestyle="--",linewidth=1)
ax1.axhline(0.518,color="orange",linestyle=":",linewidth=1.5,label="Solar rocky mean")
ax1.set_xscale("log")
ax1.set_xlabel("Orbital Distance (AU)")
ax1.set_ylabel("CCR")
ax1.set_title("H_UNIVERSAL CCR Gradient  rho="+str(round(rho,3))+"  p="+str(round(pval,4))+"  N="+str(len(an)))
ax1.legend(fontsize=8)
ax1.grid(True,alpha=0.2)
ax2.boxplot([an[an["obs_geometry"]==g]["CCR"].values for g in geoms],labels=geoms,patch_artist=True)
ax2.axhline(0.0,color="black",linestyle="--")
ax2.axhline(0.518,color="orange",linestyle=":",linewidth=1.5,label="Solar rocky CCR")
ax2.set_ylabel("CCR")
ax2.set_title("CCR by Geometry")
ax2.legend(fontsize=8)
ax2.grid(True,alpha=0.2,axis="y")
plt.tight_layout()
plt.savefig(OUTPUT_PLOT,dpi=300)
plt.close()
log("Plot saved: "+OUTPUT_PLOT)

an["h_universal_class"]=pd.cut(an["CCR"],bins=[-np.inf,0.10,0.35,np.inf],labels=["CAPTURE","GRAY_ZONE","DISK"])
cols=["pl_name","hostname","pl_orbsmax","obs_geometry","C/O","e_C/O","E_C/O","C_O_star","sigma_star","CCR","sigma_CCR","h_universal_class"]
an[cols].to_csv(OUTPUT_CSV,index=False)
log("Results saved: "+OUTPUT_CSV)

log("=== SUMMARY ===")
for cls in ["CAPTURE","GRAY_ZONE","DISK"]:
    s=an[an["h_universal_class"]==cls]
    v=round(float(s["CCR"].mean()),3) if len(s)>0 else 0
    log(cls+"  N="+str(len(s))+"  CCR_mean="+str(v))
log("Exoplanet CCR mean="+str(round(float(an["CCR"].mean()),3)))
log("Solar rocky CCR=0.518  Gap="+str(round(0.518-float(an["CCR"].mean()),3))+" dex")
if pval<0.001:
    log("GRADIENT: HIGHLY SIGNIFICANT")
elif pval<0.05:
    log("GRADIENT: SIGNIFICANT")
else:
    log("GRADIENT: NOT SIGNIFICANT p="+str(round(pval,4)))
log("DONE")
