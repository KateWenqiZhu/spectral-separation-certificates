# KE-02 实验结果索引（截至 2026-11-02）

全部文件都在 `Open question NLA100\KE02\` 下。打包文件：`KE02_results.zip`（含 paper / numerics / notes 全部）。

## 一、论文与文字稿

| 路径 | 内容 |
|---|---|
| `paper/main.pdf` / `main.tex` | 完整论文（23 页）。Section 5 Numerical experiments、Section 6 Outlook 为第一轮新写 |
| `paper/main_previous.pdf` | 第一轮改动前的 PDF（对照用） |
| `paper/section5_numerics.tex` `section6_outlook.tex` | 新增两节源文件 |
| `paper/table_sweep.tex` `table_arb.tex` `table_baselines.tex` | 正文三张表（自动生成） |
| `paper/ke02_*.pdf` | 正文四张图 |
| `notes/research_note_for_Yuji.pdf` / `.tex` | 给 Yuji 的 2 页讨论稿 |
| `README_KE02.md` | 第一轮总说明 |
| `numerics/AUDIT_round2.md` | **第二轮代码审核**：审稿人三处重点的处理与验证 |

## 二、代码（`numerics/`，可复现）

| 文件 | 作用 |
|---|---|
| `ke02num/core.py` | 六个矩阵族、F(t)、Algorithm 1、双精度 backend、arb dense backend |
| `ke02num/hybrid.py` | **结构化严格 O(n²) oracle**：dstemr + 纯 Sturm 两侧 bracketing（精确矩阵上认证）+ bisection refine + 直接 gap 包络 |
| `run_sweep.py` | 第一轮实验 1–3（双精度 sweep）→ `sweep.jsonl` |
| `run_arb.py` | 第一轮 arb dense 严格 full-K → `arb_*.json` |
| `make_figures.py` | 生成图表 → `figures/` `tables/` |
| `run_direct_s.py` | 第二轮实验 A（direct-S 模型误差版）+ hybrid → `directS_fp.jsonl` `directS_hyb.jsonl` |
| `run_adaptive.py` | 第二轮主实验：双精度选路径 + 严格 hybrid 认证 + 自适应精度 → `adaptive.jsonl` |
| `run_fallback.py` | 第二轮安全网：严格 full-K（自适应精度、保证比较）demo |
| `requirements.txt` | numpy scipy matplotlib pandas python-flint |

## 三、数据（`numerics/data/`）

| 文件 | 行数 | 内容 | n 范围 |
|---|---|---|---|
| `sweep.jsonl` | 610 | 第一轮双精度 sweep，6 族 × 30 seeds，实验 1/2/3 全部指标 | 10–200 |
| `arb_*.json` | 24 | 第一轮 arb dense 严格 full-K（每族一例） | 10–100 |
| `fp_calibration.json` | — | 双精度先验误差模型校准（最大比值 0.59） | — |
| `directS_fp.jsonl` | 694 | 实验 A：direct-S vs 1/8 停机（模型误差），6 族 × seeds | 10–1000 |
| `directS_hyb.jsonl` | 24 | hybrid 严格 direct-S 单例 | 100–1000 |
| **`adaptive.jsonl`** | 20 | **第二轮主结果（重写后严格版）**：j\*、两阶段耗时、slope cert、direct gap | 50,100,200,500(2 族) |
| `adaptive_prewrite.jsonl` | 33 | 重写前的 adaptive（含 1000/2000）。50/100/200 已验证与重写版逐位一致；500/1000/2000 的 j\*/S/cert/gap_direct 数值有效，仅严格性基础与耗时不同 | 50–2000 |

### 口径说明
- `adaptive.jsonl` 是最终严格版，但 n=500 只跑完 2 个族、1000/2000 未跑完（容器夜间闲置被停）。
- 完整 50–2000 的数值结论看 `adaptive_prewrite.jsonl`（数值一致，见上）。要补严格版的 500(其余4族)/1000/2000：`python run_adaptive.py 500 1000 2000`，再按需 `make_figures.py`。

## 四、第二轮核心结论（一句话）
- direct-S 早停把证书比 full-K 抬高 10⁷–10¹⁴ 倍；hybrid O(n²) 严格认证把 n=100 从 7–8 分钟降到 <0.1s；同时输出的 direct gap 证书（10⁻⁵ 量级）比 slope 证书（10⁻³⁴）高约 10²⁸ 倍。细节见 `AUDIT_round2.md` 与各 jsonl。
