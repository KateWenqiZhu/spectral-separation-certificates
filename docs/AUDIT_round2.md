# KE-02 numerics — 第二轮代码审核（审稿人三处重点）

日期 2026-11-01。只改了 `numerics/`，论文 PDF 未动。

审稿人要求重点检查三处。结论：三处都已处理并有独立验证。

## 1. Double-precision selection 与 full-K theorem 的关系 —— 已补严格 fallback

- 早停（direct-S）通过时，证书只依赖最终 (t,h) 处严格的 `S̄_h(t)<1/2`，与中途选窗方式无关 —— 这一点原本就成立。
- 但若早停从未成功、要依赖 Section 4 的 full-K 定理，则定理证明要求每个 endpoint-slope 的误差 ≤ 1/64（保证 inexact contraction `d_{j+1} ≤ ½d_j + 1/32`），双精度在 n≳100 做不到。
- 新增 `run_fallback.py`：full-K 路径的每个 endpoint-slope difference 和每次候选比较都用 hybrid oracle 的**严格包络 + 自适应精度**（把特征值包络用 Sturm bisection 收紧到足以判定比较，或识别为 ≤1/64 的 near-tie）。near-tie 时任选一个候选都保持 contraction（引理保证至少一个候选 ≤ ½ parent）。
- 验证（on demand，benchmark 从不触发）：

  | | d_K 严格上界 | endpoint 比较最大不确定度 | near-ties | S_h(t) 严格上界 |
  |---|---|---|---|---|
  | random n=50 | ~0 (±9e-14) <1/8 ✓ | 9.4e-14 ≤1/64 ✓ | 0 | ~0 <1/2 ✓ |
  | wilkinson n=50 | ~0 <1/8 ✓ | 8.8e-14 ✓ | 0 | ~0 ✓ |
  | random n=100 | ~0 <1/8 ✓ | 6.6e-12 ✓ | 1（已正确处理） | ~0 ✓ |

  成本高（n=50 约 27s，n=100 约 130s，因为每轮都把整谱精化），仅作安全网。

## 2. Hybrid eigenvalue enclosures 的 completeness —— 重写为纯 Sturm 认证

**原实现的弱点**（审稿人点中了）：singleton 区间的严格性唯一依赖手推的 double-precision 残差界 `r`（Krylov–Weinstein），只有 m>1 的 cluster 用 arb Sturm count 验证。实测 `r` 的最坏安全裕度只有 14%（err/width 最大 0.86）。

**重写**（`ke02num/hybrid.py:_bracket_sturm`）：每个有序特征值 λ_j 用**两侧 Sturm bracketing** 认证 —— 在**精确矩阵**（exact arb 对角 `self._cur` + 精确 β²）上验证 `count(L_j)=j`、`count(H_j)=j+1`；无法分离的（真/近重特征值）归为 cluster，用两端点 count 认证整簇恰含 m 个。双精度残差 `r` 只用来**播种**搜索起点：若 `r` 偏小，count 不匹配 → 加宽重试，**绝不会静默给出错误对应**（fail-safe）。

验证（所有族 + 边界情形）：

- vs mpmath（40 dps，n=40 全族，含 t=0 的 clustered 情形）：**0 violations**。
- 独立 400-bit Sturm 重算每个 bracket：**0 failures**。
- 零非对角边：diag_repeated（β≡0，簇大小 4）✓；单个 β_i=0 的部分可约矩阵 ✓；两个完全相同的解耦块（**精确跨块重特征值**）→ 正确检出簇大小 2，且 `gap_direct` 正确报负值（诚实表示无法认证正 gap）。
- cluster count、跨 cluster 覆盖、outer bounds（count(−2)=0, count(2)=n）全部对**精确矩阵**核验。

成本：2n 次 Sturm count（各 O(n)）= O(n²)。cheap eigs：n=500 约 1.4s，n=1000 约 6.3s（旧的信任浮点残差版本约 0.15–0.5s）。这是换取完全严格的代价；仍远低于 dense arb（n=100 要 7–8 分钟）。主结果（j*, S, cert, gap_direct）与重写前**逐位一致**。

## 3. Toeplitz 的 Stage 3 需要 bisection fallback —— 已加

**原实现**：`refine()` 只用 Newton + Sturm 验证；Newton 在 Toeplitz 两端密集特征值上收敛失败时，保留 ~5e-15 的宽区间 → n=1000 Toeplitz 的 `S=0.143±0.26`（虽仍 <0.5 可认证，但勉强）。

**修复**（`ke02num/hybrid.py:refine`）：改为**保证收敛的 Sturm bisection**（Newton 只作首步加速，之后二分；每步都是精确矩阵上的 Sturm count，故与 split 点如何选无关都严格）。真/近重特征值共享簇 bracket 的不narrow（诚实）。

效果：

| | 原 refine wmax | 新 refine wmax | 100% narrowed | 新 S 误差 |
|---|---|---|---|---|
| Toeplitz n=500 | 部分失败 | 8.1e-28 | 是 | S=222 ± 2.9e-15 |
| Toeplitz n=1000 | 部分失败（S±0.26） | 8.1e-28 | 是 | S=3548 ± 1.2e-14 |
| Toeplitz n=200（adaptive 实跑） | S=0.018±2.2e-4 | — | 是 | **S=0.310 ± 6.9e-15** |

## 附：凸性自检

`slope_variation` 里保留了「S + err < 0 即 raise」的凸性自检，作为 debugging control。审稿人说得对：它不能代替 enclosure 正确性证明 —— 后者现在由第 2 点的 Sturm 认证 + 独立重算提供。

## 文件

- `ke02num/hybrid.py`：`_bracket_sturm`（纯 Sturm 认证）、`refine`（bisection fallback）、`gap_direct_lower[_refined]`。
- `run_fallback.py`：严格 full-K 安全网 demo。
- 其余（`run_adaptive.py`、`run_direct_s.py`、`core.py`）不变。
