"""
关灯科技实战任务 - 拣货缺货拦截分析
根问题：库存同步时滞导致账面库存失真，拣货员空跑。
核心结论：用「库存同步时滞 > 10 分钟」拦截疑似缺货任务，
        可净节省 9280 分钟工时（挽回约 62% 的缺货空跑损失）。

口径：每次到位缺货损失 90 分钟；每提前拦截一次花费 10 分钟。
净收益 = 拦对(真缺货) * (90-10) - 拦错(其实是成功) * 10
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体（Windows 自带），避免图表中文乱码
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

CSV_PATH = "pick_tasks.csv"

# ---------- 1. 读取与特征构造 ----------
df = pd.read_csv(CSV_PATH)
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
df["stock_synced_at"] = pd.to_datetime(df["stock_synced_at"], errors="coerce")
# 时滞（分钟）：任务下发时间 - 账面库存最后同步时间
df["lag_min"] = (df["created_at"] - df["stock_synced_at"]).dt.total_seconds() / 60

# ---------- 2. 标签与缺货率 ----------
df["f"] = pd.to_numeric(df["is_fulfilled"], errors="coerce")
known = df[df["f"].isin([0, 1])].copy()
missing = known[known["f"] == 0]
success = known[known["f"] == 1]
n_miss = len(missing)
baseline_loss = n_miss * 90  # 不拦截的基线损失

print("===== 数据概况 =====")
print(f"总任务 {len(df)} | 有结果 {len(known)} | 未回传 {len(df)-len(known)}")
print(f"缺货 {n_miss} | 成功 {len(success)} | 缺货率 {n_miss/len(known):.1%}")

# ---------- 3. 特征区分度（中位数对比）----------
print("\n===== 特征区分度（中位数）=====")
for col in ["lag_min", "book_stock", "aisle_traffic"]:
    print(f"{col:>12} | 缺货 {missing[col].median():7.2f} | 成功 {success[col].median():7.2f}")

# ---------- 4. 品类 / 门店缺货率 ----------
print("\n===== 品类缺货率 =====")
c = known.groupby("category").agg(总=("f", "size"), 缺货=("f", lambda x: (x == 0).sum()))
c["缺货率"] = c["缺货"] / c["总"]
print(c.sort_values("缺货率", ascending=False).to_string())

print("\n===== 门店缺货率（缺货数>=5）=====")
s = known.groupby("store_name").agg(总=("f", "size"), 缺货=("f", lambda x: (x == 0).sum()))
s["缺货率"] = s["缺货"] / s["总"]
print(s[s["缺货"] >= 5].sort_values("缺货率", ascending=False).to_string())

# ---------- 5. 拦截规则回测 ----------
def evaluate(pred):
    tp = ((pred == 1) & (known["f"] == 0)).sum()
    fp = ((pred == 1) & (known["f"] == 1)).sum()
    fn = n_miss - tp
    net = tp * 80 - fp * 10
    return dict(召回率=tp / n_miss, 精确率=tp / (tp + fp) if (tp + fp) else 0,
                净省分钟=net, 拦对tp=tp, 拦错fp=fp, 漏拦fn=fn)

print("\n===== 时滞阈值拦截效果（最优规则在此）=====")
best = None
for th in [6, 10, 12, 15, 20, 30, 45, 60]:
    r = evaluate((known["lag_min"] > th).astype(int))
    print(f"时滞>{th:>2}分钟 -> 召回{r['召回率']:.0%} 精确{r['精确率']:.0%} "
          f"净省{r['净省分钟']:.0f}分")
    if best is None or r["净省分钟"] > best["净省分钟"]:
        best = dict(阈值=th, **r)

print(f"\n>> 最优规则：时滞 > {best['阈值']} 分钟")
print(f"   召回率 {best['召回率']:.0%} | 精确率 {best['精确率']:.0%} | "
      f"净省 {best['净省分钟']:.0f} 分钟")
print(f"   基线损失 {baseline_loss} 分钟 -> 挽回 {best['净省分钟']/baseline_loss:.0%}")

# 对比：book_stock 规则（证明其无效）
print("\n===== book_stock 阈值（对比，几乎无区分度）=====")
for th in [1, 3, 5]:
    r = evaluate((known["book_stock"] <= th).astype(int))
    print(f"book_stock<={th} -> 召回{r['召回率']:.0%} 精确{r['精确率']:.0%} "
          f"净省{r['净省分钟']:.0f}分")

# ---------- 6. 出图 ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

axes[0].hist([missing["lag_min"], success["lag_min"]], bins=30,
             label=["缺货", "成功"], alpha=0.6, range=(0, 60))
axes[0].axvline(best["阈值"], color="red", linestyle="--", label=f"拦截阈值 {best['阈值']} 分钟")
axes[0].set_title("库存同步时滞分布（分钟）")
axes[0].set_xlabel("时滞(分钟)")
axes[0].legend()

c["缺货率"].sort_values().plot(kind="barh", ax=axes[1], color="steelblue")
axes[1].set_title("各品类缺货率")

s2 = s.sort_values("缺货率", ascending=False)
axes[2].bar(s2.index, s2["缺货率"], color="coral")
axes[2].set_title("各门店缺货率")
axes[2].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("charts_overview.png", dpi=120)
print("\n图表已保存: charts_overview.png")
print("完成。")
