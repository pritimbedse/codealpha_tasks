"""
Unemployment in India (May 2019 - Jun 2020)
Data cleaning, exploration, visualisation, Covid-19 impact, seasonality, policy insights.

Usage:
    python unemployment_analysis.py                          # looks for "Unemployment in India.csv" next to the script
    python unemployment_analysis.py --data path/to/file.csv --out outputs

Dataset columns (the CSV you downloaded has NO header row, so names are assigned here):
    Region, Date, Frequency, Unemployment_Rate (%), Employed, Labour_Participation_Rate (%), Area
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

COLS = ["Region", "Date", "Frequency", "Unemployment_Rate",
        "Employed", "Labour_Participation_Rate", "Area"]

ZONES = {
    "North": ["Delhi", "Haryana", "Himachal Pradesh", "Jammu & Kashmir", "Punjab",
              "Rajasthan", "Uttar Pradesh", "Uttarakhand", "Chandigarh"],
    "South": ["Andhra Pradesh", "Karnataka", "Kerala", "Tamil Nadu", "Telangana", "Puducherry"],
    "East": ["Bihar", "Jharkhand", "Odisha", "West Bengal"],
    "West": ["Goa", "Gujarat", "Maharashtra"],
    "Central": ["Chhattisgarh", "Madhya Pradesh"],
    "North-East": ["Assam", "Meghalaya", "Sikkim", "Tripura"],
}
REGION_TO_ZONE = {r: z for z, regs in ZONES.items() for r in regs}

LOCKDOWN_MONTH = pd.Timestamp("2020-03-01")   # India's national lockdown began 25 Mar 2020
COVID_MONTH = pd.Timestamp("2020-04-01")      # first full lockdown month

sns.set_theme(style="whitegrid", context="notebook")
C_MAIN, C_RURAL, C_URBAN, C_COVID = "#1f77b4", "#2ca02c", "#d62728", "#999999"


# --------------------------------------------------------------------------- #
# 1. LOAD + CLEAN
# --------------------------------------------------------------------------- #
def load_and_clean(path):
    raw = pd.read_csv(path, header=None, names=COLS, encoding="utf-8-sig",
                      skipinitialspace=True, dtype=str)
    report = {"raw_rows": len(raw)}

    # fully blank rows (this dataset has some)
    raw = raw.dropna(how="all")
    report["blank_rows_removed"] = report["raw_rows"] - len(raw)

    for c in ["Region", "Frequency", "Area"]:
        raw[c] = raw[c].str.strip()

    # dates are dd-mm-yyyy; anything unparseable (e.g. a header row) is dropped
    raw["Date"] = pd.to_datetime(raw["Date"].str.strip(), format="%d-%m-%Y", errors="coerce")
    bad = raw["Date"].isna().sum()
    raw = raw.dropna(subset=["Date"])
    report["bad_date_rows_removed"] = int(bad)

    for c in ["Unemployment_Rate", "Employed", "Labour_Participation_Rate"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    report["rows_with_missing_values"] = int(raw.isna().any(axis=1).sum())
    raw = raw.dropna()

    before = len(raw)
    df = raw.drop_duplicates(subset=["Region", "Date", "Area"]).copy()
    report["duplicates_removed"] = before - len(df)

    # derived columns
    df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    df["Month_Num"] = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.strftime("%b")
    df["Zone"] = df["Region"].map(REGION_TO_ZONE).fillna("Other")
    df["Period"] = np.where(df["Month"] < LOCKDOWN_MONTH, "Pre-Covid",
                            np.where(df["Month"] < COVID_MONTH, "Lockdown starts (Mar-20)", "Covid lockdown"))

    # The rate is unemployed / labour force, so we can rebuild head-counts and take
    # properly weighted averages instead of averaging percentages.
    df["Labour_Force"] = df["Employed"] / (1 - df["Unemployment_Rate"] / 100)
    df["Unemployed"] = df["Labour_Force"] - df["Employed"]

    report["clean_rows"] = len(df)
    return df.sort_values(["Region", "Area", "Date"]).reset_index(drop=True), report


def balanced_panel(df):
    """Keep only Region+Area series observed in every month, so month-to-month changes
    are not caused by series entering/leaving the sample."""
    n_months = df["Month"].nunique()
    counts = df.groupby(["Region", "Area"])["Month"].transform("nunique")
    return df[counts == n_months].copy()


def weighted_rate(frame, by):
    g = frame.groupby(by)[["Unemployed", "Labour_Force", "Employed"]].sum()
    g["Rate"] = g["Unemployed"] / g["Labour_Force"] * 100
    return g


def shade_covid(ax):
    ax.axvspan(COVID_MONTH - pd.Timedelta(days=15), df_end + pd.Timedelta(days=15),
               color=C_COVID, alpha=0.15, label="Covid lockdown")


# --------------------------------------------------------------------------- #
# 2. MAIN ANALYSIS
# --------------------------------------------------------------------------- #
def main():
    global df_end
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--data", default=os.path.join(here, "Unemployment in India.csv"))
    ap.add_argument("--out", default=os.path.join(here, "outputs"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df, rep = load_and_clean(args.data)
    df_end = df["Month"].max()
    pan = balanced_panel(df)
    lines = []  # collected findings -> findings.txt

    def say(text=""):
        print(text)
        lines.append(text)

    # ---------------- data quality ----------------
    say("=" * 70)
    say("DATA CLEANING REPORT")
    say("=" * 70)
    for k, v in rep.items():
        say(f"{k:28s}: {v}")
    say(f"Period covered              : {df['Month'].min():%b %Y} to {df['Month'].max():%b %Y} "
        f"({df['Month'].nunique()} months)")
    say(f"Regions / Areas             : {df['Region'].nunique()} / {sorted(df['Area'].unique())}")
    say(f"Balanced series used for national trends: "
        f"{pan.groupby(['Region', 'Area']).ngroups} of {df.groupby(['Region', 'Area']).ngroups}")
    say("\nRows per month (coverage is slightly uneven):")
    say(df.groupby("Month").size().rename(lambda d: f"{d:%b-%Y}").to_string())

    say("\nSummary statistics:")
    say(df[["Unemployment_Rate", "Employed", "Labour_Participation_Rate"]].describe().round(2).to_string())

    # ---------------- national trend ----------------
    nat = weighted_rate(pan, "Month")
    nat_area = weighted_rate(pan, ["Month", "Area"]).reset_index()
    pre_months = nat.index < LOCKDOWN_MONTH
    pre_rate = nat.loc[pre_months, "Rate"].mean()

    say("\n" + "=" * 70)
    say("NATIONAL TREND (employment-weighted, balanced panel)")
    say("=" * 70)
    say(nat["Rate"].round(2).rename(lambda d: f"{d:%b-%Y}").to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(nat.index, nat["Rate"], marker="o", color=C_MAIN, lw=2.5)
    shade_covid(ax)
    ax.set(title="National unemployment rate", ylabel="Unemployment rate (%)")
    ax.legend()
    ax = axes[1]
    for area, col in [("Rural", C_RURAL), ("Urban", C_URBAN)]:
        s = nat_area[nat_area["Area"] == area]
        ax.plot(s["Month"], s["Rate"], marker="o", color=col, lw=2.5, label=area)
    shade_covid(ax)
    ax.set(title="Rural vs Urban", ylabel="Unemployment rate (%)")
    ax.legend()
    for a in axes:
        a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b-%y"))
        a.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig1_national_trend.png"), dpi=150)
    plt.close(fig)

    # ---------------- Covid impact (national) ----------------
    say("\n" + "=" * 70)
    say("COVID-19 IMPACT")
    say("=" * 70)
    peak_month = nat.loc[nat.index >= COVID_MONTH, "Rate"].idxmax()
    apr, peak, last = nat.loc[COVID_MONTH, "Rate"], nat.loc[peak_month, "Rate"], nat["Rate"].iloc[-1]
    say(f"Pre-Covid average ({nat.index.min():%b-%y} to Feb-20): {pre_rate:.2f}%")
    say(f"Peak during Covid: {peak:.2f}% in {peak_month:%b-%Y}  (x{peak / pre_rate:.1f} the pre-Covid level)")
    say(f"Latest month ({nat.index[-1]:%b-%Y}): {last:.2f}%")

    for area in ["Rural", "Urban"]:
        s = nat_area[nat_area["Area"] == area].set_index("Month")["Rate"]
        say(f"{area:5s}: pre-Covid {s[s.index < LOCKDOWN_MONTH].mean():.2f}%  ->  "
            f"peak {s[s.index >= COVID_MONTH].max():.2f}% "
            f"({s[s.index >= COVID_MONTH].idxmax():%b-%y})  ->  latest {s.iloc[-1]:.2f}%")

    # jobs and participation
    emp_idx = nat["Employed"] / nat.loc[pre_months, "Employed"].mean() * 100
    lpr = pan.groupby("Month")["Labour_Participation_Rate"].mean()
    say(f"\nEmployed persons (balanced panel), index pre-Covid avg = 100:")
    say(emp_idx.round(1).rename(lambda d: f"{d:%b-%Y}").to_string())
    say(f"Lowest employment index: {emp_idx.min():.1f} in {emp_idx.idxmin():%b-%Y} "
        f"(~{100 - emp_idx.min():.0f}% fewer employed than the pre-Covid average)")
    say(f"Labour participation: pre-Covid avg {lpr[lpr.index < LOCKDOWN_MONTH].mean():.1f}% -> "
        f"low {lpr.min():.1f}% in {lpr.idxmin():%b-%Y}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(emp_idx.index, emp_idx, marker="o", color=C_MAIN, lw=2.5)
    axes[0].axhline(100, color="black", ls="--", lw=1)
    axes[0].set(title="Employed persons (index, pre-Covid average = 100)", ylabel="Index")
    axes[1].plot(lpr.index, lpr, marker="o", color=C_URBAN, lw=2.5)
    axes[1].set(title="Labour participation rate (mean across series)", ylabel="%")
    for a in axes:
        shade_covid(a)
        a.legend()
        a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b-%y"))
        a.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig2_employment_participation.png"), dpi=150)
    plt.close(fig)

    # distribution before vs during
    fig, ax = plt.subplots(figsize=(9, 5))
    order = ["Pre-Covid", "Lockdown starts (Mar-20)", "Covid lockdown"]
    sns.boxplot(data=df, x="Period", y="Unemployment_Rate", order=order, hue="Area",
                palette={"Rural": C_RURAL, "Urban": C_URBAN}, ax=ax)
    ax.set(title="Distribution of state-level unemployment rates", ylabel="Unemployment rate (%)", xlabel="")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig3_distribution_by_period.png"), dpi=150)
    plt.close(fig)

    # ---------------- state level ----------------
    st = weighted_rate(df, ["Region", "Month"])["Rate"].unstack("Month")
    pre_cols = [c for c in st.columns if c < LOCKDOWN_MONTH]
    cov_cols = [c for c in st.columns if c >= COVID_MONTH]
    impact = pd.DataFrame({
        "Pre_Covid_avg": st[pre_cols].mean(axis=1),
        "Peak_Covid": st[cov_cols].max(axis=1),
        "Latest": st.iloc[:, -1],
    })
    impact["Increase_pp"] = impact["Peak_Covid"] - impact["Pre_Covid_avg"]
    impact["Recovered_pp"] = impact["Peak_Covid"] - impact["Latest"]
    impact = impact.sort_values("Increase_pp", ascending=False)
    impact.round(2).to_csv(os.path.join(args.out, "state_covid_impact.csv"))

    say("\nTop 10 states by rise in unemployment (pre-Covid average -> Covid peak, percentage points):")
    say(impact.head(10).round(1).to_string())
    say(f"\nStates where unemployment rose: {(impact['Increase_pp'] > 0).sum()} of {len(impact)}")
    say(f"Lowest rises: {', '.join(impact.tail(3).index)}")

    top = impact.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(top))
    ax.barh(y + 0.2, top["Pre_Covid_avg"], 0.4, color=C_MAIN, label="Pre-Covid average")
    ax.barh(y - 0.2, top["Peak_Covid"], 0.4, color=C_URBAN, label="Covid peak")
    ax.set_yticks(y, top.index)
    ax.set(title="10 states hit hardest by Covid-19 (unemployment rate, %)", xlabel="%")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig4_state_covid_impact.png"), dpi=150)
    plt.close(fig)

    # heatmap
    hm = st.copy()
    hm.columns = [f"{c:%b-%y}" for c in hm.columns]
    hm = hm.loc[impact.index]
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(hm, cmap="YlOrRd", annot=False, cbar_kws={"label": "Unemployment rate (%)"}, ax=ax)
    ax.set(title="Unemployment rate by state and month (sorted by Covid impact)", xlabel="", ylabel="")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig5_state_month_heatmap.png"), dpi=150)
    plt.close(fig)

    # structurally high / low states before Covid
    pre_rank = impact["Pre_Covid_avg"].sort_values(ascending=False)
    say("\nHighest pre-Covid unemployment (structural problem):")
    say(pre_rank.head(5).round(1).to_string())
    say("Lowest pre-Covid unemployment:")
    say(pre_rank.tail(5).round(1).to_string())

    # ---------------- zones ----------------
    zone = weighted_rate(df, ["Zone", "Month"])["Rate"].unstack("Zone")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for z in zone.columns:
        ax.plot(zone.index, zone[z], marker="o", lw=2, label=z)
    shade_covid(ax)
    ax.set(title="Unemployment rate by zone", ylabel="Unemployment rate (%)")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b-%y"))
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig6_zones.png"), dpi=150)
    plt.close(fig)
    zsum = pd.DataFrame({
        "Pre_Covid_avg": zone[zone.index < LOCKDOWN_MONTH].mean(),
        "Peak_Covid": zone[zone.index >= COVID_MONTH].max(),
    })
    zsum["Increase_pp"] = zsum["Peak_Covid"] - zsum["Pre_Covid_avg"]
    say("\nBy zone:")
    say(zsum.sort_values("Increase_pp", ascending=False).round(1).to_string())

    # ---------------- seasonality ----------------
    say("\n" + "=" * 70)
    say("SEASONALITY / PATTERNS")
    say("=" * 70)
    pre = nat[pre_months]
    say(f"Pre-Covid national rate ranged {pre['Rate'].min():.2f}% - {pre['Rate'].max():.2f}% "
        f"(std {pre['Rate'].std():.2f} pp); highest in {pre['Rate'].idxmax():%b-%Y}, "
        f"lowest in {pre['Rate'].idxmin():%b-%Y}.")
    say("NOTE: the data holds only one pre-Covid year, so each calendar month appears once. "
        "Month-of-year patterns below are indicative, not statistically proven seasonality.")

    # same-month comparison: May and June 2019 vs 2020
    yoy = nat.loc[[m for m in nat.index if m.month in (5, 6)], "Rate"]
    yoy_tbl = pd.DataFrame({"Year": yoy.index.year, "Month": yoy.index.strftime("%b"), "Rate": yoy.values})
    say("\nSame calendar month, year on year (removes seasonality from the Covid comparison):")
    say(yoy_tbl.pivot(index="Month", columns="Year", values="Rate").round(2).to_string())

    # state-level volatility before Covid
    vol = st[pre_cols].std(axis=1).sort_values(ascending=False)
    say("\nMost volatile states before Covid (std dev of monthly rate, pp):")
    say(vol.head(5).round(2).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.bar(pre.index.strftime("%b-%y"), pre["Rate"], color=C_MAIN)
    ax.axhline(pre_rate, color="black", ls="--", lw=1, label=f"Average {pre_rate:.1f}%")
    ax.set(title="Pre-Covid monthly pattern (one year only)", ylabel="Unemployment rate (%)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    pv = yoy_tbl.pivot(index="Month", columns="Year", values="Rate").reindex(["May", "Jun"])
    pv.plot(kind="bar", ax=axes[1], color=[C_MAIN, C_URBAN], rot=0)
    axes[1].set(title="May and June: 2019 vs 2020", ylabel="Unemployment rate (%)", xlabel="")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig7_seasonality.png"), dpi=150)
    plt.close(fig)

    # ---------------- correlation ----------------
    corr = df[["Unemployment_Rate", "Employed", "Labour_Participation_Rate"]].corr()
    say("\nCorrelation matrix:")
    say(corr.round(2).to_string())

    # ---------------- save ----------------
    df.to_csv(os.path.join(args.out, "unemployment_cleaned.csv"), index=False)
    nat.round(3).to_csv(os.path.join(args.out, "national_monthly.csv"))
    with open(os.path.join(args.out, "findings.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved cleaned data, tables, findings.txt and 7 charts to: {args.out}")


if __name__ == "__main__":
    main()