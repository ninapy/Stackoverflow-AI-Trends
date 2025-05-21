# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
import datetime as dt
# %%
questions_counts_per_month = pd.read_csv("repo/final-projects-stackoverflow-analysis-group/data/datasets/question_counts.csv")
questions_counts_per_month

# %%
import matplotlib.dates as mdates
try:
    df = questions_counts_per_month.copy()
except NameError:
    df = pd.read_csv(
        "repo/final-projects-stackoverflow-analysis-group/data/datasets/question_counts.csv"
    )
    print("CSV re-loaded → df.shape =", df.shape)

df["PostMonth"] = pd.to_datetime(df["PostMonth"], format="%Y-%m")
df.sort_values("PostMonth", inplace=True)
cutoff = pd.Timestamp("2022-11-01")      # ChatGPT public release
df = df[df["PostMonth"] >= pd.Timestamp("2021-01-01")]
pre  = df[df["PostMonth"] <  cutoff]
post = df[df["PostMonth"] >= cutoff]
t_stat, p_val = stats.ttest_ind(
    pre["QuestionCount"],
    post["QuestionCount"],
    equal_var=False,
    nan_policy="omit",
)
pre_mean  = pre["QuestionCount"].mean()
post_mean = post["QuestionCount"].mean()
pct_change = 100 * (post_mean - pre_mean) / pre_mean
plt.style.use("ggplot")
sns.set_theme(context="talk")

fig, (ax_ts, ax_box) = plt.subplots(
    2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1]}
)

ax_ts.plot(df["PostMonth"], df["QuestionCount"], marker="o", lw=1)
ax_ts.axvline(cutoff, color="red", ls="--", label="ChatGPT Release\n(2022-11)")
ax_ts.set(
    title="Stack Overflow Questions Per Month",
    xlabel="Date", ylabel="Question Count"
)
ax_ts.xaxis.set_major_locator(mdates.YearLocator())
ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax_ts.legend()

sns.boxplot(
    x="Period", y="QuestionCount",
    data=df, palette=["#3b8bba", "#ba3b3b"],
    ax=ax_box
)
ax_box.set(
    title="Distribution Before vs. After ChatGPT",
    xlabel="", ylabel="Question Count"
)

txt  = (
    f"Pre-ChatGPT mean:  {pre_mean:,.0f}\n"
    f"Post-ChatGPT mean: {post_mean:,.0f}\n"
    f"Δ % change:        {pct_change:+.2f}%\n"
    f"t-stat = {t_stat:0.2f}    p-value = {p_val:.4g}"
)
props = dict(boxstyle="round", facecolor="white", alpha=0.8)
ax_box.text(
    0.5, 0.95, txt, transform=ax_box.transAxes,
    ha="center", va="top", bbox=props, fontsize=12
)

plt.tight_layout()
plt.show()

print("────────────────────────  t-test summary  ────────────────────────")
print(f"Pre-ChatGPT  (n={len(pre)}):  μ = {pre_mean:,.0f}")
print(f"Post-ChatGPT (n={len(post)}): μ = {post_mean:,.0f}")
print(f"Percent change: {pct_change:+.2f}%")
print(f"t-statistic   : {t_stat:0.2f}")
print(f"p-value       : {p_val:.4g}")
alpha = 0.05
print("Statistically significant?" , "YES" if p_val < alpha else "NO")
plt.savefig('monthly_posts_t-test.png')

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

cutoff = pd.Timestamp("2022-11-01") 
tmp = df.copy()
tmp["PostMonth"] = pd.to_datetime(tmp["PostMonth"])
tmp.sort_values("PostMonth", inplace=True)

days_per_month = 30.437         
tmp["t_mon"]   = (tmp["PostMonth"] - tmp["PostMonth"].min()).dt.days / days_per_month
tmp["post"]    = (tmp["PostMonth"] >= cutoff).astype(int)
tmp["t_post"]  = tmp["t_mon"] * tmp["post"]  

formula = "QuestionCount ~ t_mon + post + t_post"
model   = smf.ols(formula, data=tmp).fit()

β0, β1, β2, β3 = model.params
pre_slope   = β1                   #  questions / month BEFORE Nov-2022
post_slope  = β1 + β3              #  questions / month AFTER  Nov-2022
delta_slope = β3                   #  acceleration (post − pre)

print("\n──────────────── Interrupted time-series regression ────────────────")
print(f"Formula      : {formula}")
print(f"Breakpoint   : {cutoff:%Y-%m}")
print(model.summary().tables[1])   # coefficient table

print("\nDerived monthly slopes (questions / month)")
print(f"  Pre-ChatGPT slope  : {pre_slope:>9.1f}")
print(f"  Post-ChatGPT slope : {post_slope:>9.1f}")
print(f"  Δ slope (post-pre) : {delta_slope:>9.1f}")
print("-------------------------------------------------------------------\n")
sns.set_theme(style="whitegrid", context="talk")
fig, ax = plt.subplots(figsize=(14, 5))
ax.scatter(tmp["PostMonth"], tmp["QuestionCount"], s=35, alpha=.6,
           label="Observed")
tmp["pred"] = model.predict(tmp)
ax.plot(tmp["PostMonth"], tmp["pred"], color="black", lw=2, label="Fitted")
ax.axvline(cutoff, color="red", ls="--", lw=1)
ax.annotate("ChatGPT release\n(Nov-2022)", xy=(cutoff, tmp["QuestionCount"].max()*0.95),
            xytext=(6,0), textcoords="offset points", color="red", va="top")
ax.text(cutoff + pd.Timedelta(days=20),
        tmp["pred"].loc[tmp["post"]==1].iloc[0],
        f"Δ slope ≈ {delta_slope:,.0f} q / month",
        color="red")

ax.set(title="Stack Overflow Questions – Piece-wise Linear Fit",
       xlabel="Month", ylabel="Question Count")
ax.legend()
plt.tight_layout()
plt.show()

# %%
#save this 
plt.savefig('montlhy_posts_piecewise_linear_fit.png')

# %%