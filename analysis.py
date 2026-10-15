"""
关灯科技实战任务 - 拣货缺货拦截分析
根问题：账面库存(book_stock / stock_synced_at)与货架实际库存失真，
       导致"账面有货、实际缺货"，拣货员空跑。
目标：用一个低成本拦截规则，预判会缺货的任务，用 10 分钟拦截成本
       替代 90 分钟空跑损失。

口径：每次到位缺货损失 90 分钟；每提前拦截一次花费 10 分钟。
净收益 = 拦对(真缺货) * (90-10) - 拦错(其实是成功) * 10
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无 GUI 环境也能出图
import matplotlib.pyplot as plt

CSV_PATH = "pick_tasks.csv"  # 改成你本地的实际路径

# ---------- 1. 读取 ----------
df = pd.read_csv(CSV_PATH)
print("原始行数:", len(df), "| 列:", list(df.columns))

# ---------- 2. 时间解析 + 构造关键特征 ----------
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
df["stock_synced_at"] = pd.to_datetime(df["stock_synced_at"], errors="coerce")

# 时滞（小时）：任务下发时间 - 账面库存最后同步时间
df["sync_lag_hours"] = (df["created_at"] - df["stock_synced_at"]).dt.total_seconds() / 3600

# ---------- 3. 标签与缺货率 ----------
# is_fulfilled: 1=成功, 0=缺货, 空=未回传
df["is_fulfilled_num"] = pd.to_numeric(df["is_fulfilled"], errors="coerce")
known = df[df["is_fulfilled_num"].isin([0, 1])].copy()  # 只看有明确结果的
missing = known[known["is_fulfilled_num"] == 0]
success = known[known["is_fulfilled_num"] == 1]

total = len(df)
n_known = len(known)
n_missing = len(missing)
n_success = len(success)
n_noreply = total - n_known

print("\n===== 数据概况 =====")
print(f"总任务 {total} | 有结果 {n_known} | 未回传 {n_noreply}")
print(f"缺货 {n_missing} | 成功 {n_success}")
print(f"缺货率(有结果内) = {n_missing / n_known:.1%}")

# ---------- 4. 关键特征对比：缺货 vs 成功 ----------
print("\n===== 特征对比(均值) =====")
for col in ["book_stock", "sync_lag_hours", "aisle_traffic"]:
    m = missing[col].mean()
    s = success[col].mean()
    print(f"{col:>15} | 缺货 {m:8.2f} | 成功 {s:8.2f}")

# 时滞为负/缺失的比例（数据质量问题，也值得写进 README）
lag_bad = (known["sync_lag_hours"] < 0) | known["sync_lag_hours"].isna()
print(f"时滞异常/缺失占比 = {lag_bad.mean():.1%}")

# ---------- 5. 拦截规则 + 效果评估 ----------
def evaluate(pred_series, label_series):
    """pred=1 表示拦截；label=0 表示缺货"""
    tp = ((pred_series == 1) & (label_series == 0)).sum()   # 拦对(真缺货)
    fp = ((pred_series == 1) & (label_series == 1)).sum()   # 拦错(其实是成功)
    n_miss = (label_series == 0).sum()                      # 真实缺货总数
    fn = n_miss - tp                                        # 漏拦的缺货
    net_save = tp * 80 - fp * 10                            # 净节省分钟
    baseline = n_miss * 90                                  # 不拦截的基线损失
    recall = tp / n_miss if n_miss else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    return dict(TP=tp, FP=fp, FN=fn, 缺货总数=n_miss,
                召回率=recall, 精确率=prec,
                净节省分钟=net_save, 基线损失分钟=baseline)

# 规则 A：账面库存很低就拦截
print("\n===== 拦截规则效果 (在历史有结果的任务上回测) =====")
for th in [0, 1, 2, 3, 5]:
    pred = (known["book_stock"] <= th).astype(int)
    r = evaluate(pred, known["is_fulfilled_num"])
    print(f"规则A book_stock<={th:>2} -> 召回{r['召回率']:.0%} 精确{r['精确率']:.0%} "
          f"净省{r['净节省分钟']:.0f}分钟 (基线损失{r['基线损失分钟']:.0f})")

# 规则 B：时滞超过阈值就拦截
for th in [6, 12, 24, 48]:
    pred = (known["sync_lag_hours"] > th).astype(int)
    r = evaluate(pred, known["is_fulfilled_num"])
    print(f"规则B 时滞>{th:>3}h -> 召回{r['召回率']:.0%} 精确{r['精确率']:.0%} "
          f"净省{r['净节省分钟']:.0f}分钟")

# 规则 C（组合，可自行调）：账面库存低 且 时滞大
for bs, lag in [(3, 12), (5, 24), (2, 6)]:
    pred = ((known["book_stock"] <= bs) & (known["sync_lag_hours"] > lag)).astype(int)
    r = evaluate(pred, known["is_fulfilled_num"])
    print(f"规则C book<={bs} 且 时滞>{lag}h -> 召回{r['召回率']:.0%} 精确{r['精确率']:.0%} "
          f"净省{r['净节省分钟']:.0f}分钟")

# ---------- 6. 出图 ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist([missing["book_stock"], success["book_stock"]],
             bins=20, label=["缺货", "成功"], alpha=0.6)
axes[0].set_title("book_stock distribution")
axes[0].legend()

axes[1].hist([missing["sync_lag_hours"].dropna(), success["sync_lag_hours"].dropna()],
             bins=20, label=["缺货", "成功"], alpha=0.6)
axes[1].set_title("sync_lag_hours distribution")
axes[1].legend()

# 各品类缺货率
cat_rate = known.groupby("category")["is_fulfilled_num"].apply(lambda x: (x == 0).mean())
cat_rate.sort_values().plot(kind="barh", ax=axes[2])
axes[2].set_title("缺货率 by category")

plt.tight_layout()
plt.savefig("charts_overview.png", dpi=120)
print("\n图表已保存: charts_overview.png")
print("\n完成。把上面打印的关键数字填进 README 即可。")
