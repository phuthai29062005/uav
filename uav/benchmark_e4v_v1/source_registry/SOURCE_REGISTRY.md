# SOURCE_REGISTRY — E4V Reproduction-Grade DMOP Comparison

## LEVEL 1 — AUTHORITATIVE

### S1: Official CEC2018 DMOP Technical Report
- **Title**: Benchmark Problems for CEC'2018 Competition on Dynamic Multiobjective Optimisation
- **Authors**: S. Jiang, S. Yang, X. Yao, K.C. Tan, M. Kaiser, N. Krasnogor
- **Venue**: Technical Report, Newcastle University, U.K., December 2017
- **DOI/URL**: https://sites.google.com/view/shouyongjiang/resources/cec2018
- **PDF URL**: http://homepages.cs.ncl.ac.uk/shouyong.jiang/cec2018/DMOP.pdf (cert expired; cached)
- **Used for**: DF1-DF14 formulas, time model, experimental protocol, IGD/MIGD metric, variable bounds
- **Access date**: 2026-10-05
- **Status**: AUTHORITATIVE (primary benchmark specification)

### S2: Official MATLAB Implementation (author-maintained)
- **Title**: cec2018_DF.m — Official MATLAB benchmark code
- **Authors**: S. Jiang
- **URL**: http://homepages.cs.ncl.ac.uk/shouyong.jiang/cec2018/cec2018_DF.m (cert expired)
- **Used for**: Reference implementation of DF1-DF14 objective formulas
- **Access date**: 2026-10-05
- **Status**: AUTHORITATIVE (author's own code)

### S3: Official MATLAB PF Sampling Code
- **Title**: cec2018_DF_PF.m — Official PF sampling code
- **Authors**: S. Jiang
- **URL**: http://homepages.cs.ncl.ac.uk/shouyong.jiang/cec2018/cec2018_DF_PF.m (cert expired)
- **Used for**: True Pareto front generation reference
- **Access date**: 2026-10-05
- **Status**: AUTHORITATIVE (author's own code; not successfully downloaded due to expired cert)

### S4: Official C++ Implementation
- **Title**: cec2018_DF.h — Official C++ benchmark code
- **Authors**: S. Jiang
- **URL**: http://homepages.cs.ncl.ac.uk/shouyong.jiang/cec2018/cec2018_DF.h (cert expired; not downloaded)
- **Used for**: Cross-reference
- **Status**: AUTHORITATIVE (not successfully accessed)

## LEVEL 2 — INDEPENDENT REFERENCE IMPLEMENTATIONS

### S5: ahcheriet/CEC2018_DMOP Python Translation
- **Title**: CEC2018 DF Dynamic Multiobjective optimization benchmarks
- **Authors**: A.H. Cheriet (translator from official MATLAB)
- **Repository**: https://github.com/ahcheriet/CEC2018_DMOP
- **File**: cec2018_DF.py
- **Used for**: Cross-validation of DF1-DF14 formulas (primary Level 2 reference)
- **Access date**: 2026-10-05
- **Status**: INDEPENDENT_REFERENCE (explicitly translated from official MATLAB)
- **Note**: Header states "The code is translated from Matlab by ahcheriet@gmail.com"

## FORMULA REGISTRY (verified against S2 MATLAB + S6 pymoo)

| Problem | n_obj | Primary source | Verified against |
|---------|-------|----------------|-----------------|
| DF1 | 2 | S2 MATLAB | S5 match, S6 match |
| DF2 | 2 | S2 MATLAB | S5 match, S6 match |
| DF3 | 2 | S2 MATLAB | S5 ERROR (H applied to wrong term), S6 match |
| DF4 | 2 | S2 MATLAB | S5 ERROR (divide by variable not index), S6 match |
| DF5 | 2 | S2 MATLAB | S5 match, S6 match |
| DF6 | 2 | S2 MATLAB | S5 match, S6 match |
| DF7 | 2 | S2 MATLAB | S5 match, S6 match |
| DF8 | 2 | S2 MATLAB | S5 ERROR (spurious exponent + wrong variable), S6 match |
| DF9 | 2 | S2 MATLAB | S5 match, S6 match |
| DF10 | 3 | S2 MATLAB | S5 ERROR (2π not 4π), S6 match |
| DF11 | 3 | S2 MATLAB | S5 match, S6 match |
| DF12 | 3 | S2 MATLAB | S5 ERROR (missing abs, wrong var, missing x₂), S6 match |
| DF13 | 3 | S2 MATLAB | S5 match, S6 match |
| DF14 | 3 | S2 MATLAB | S5 match, S6 match |
| Time model | - | S2 MATLAB | t = (1/n_t)*floor(max(τ+τ_t-(T0+1),0)/τ_t), T0=50 |
| IGD/MIGD | - | S1 report | Standard definition |

### S6: pymoo DF Implementation (anyoptimization/pymoo)
- **Title**: pymoo dynamic DF test problems
- **Repository**: https://github.com/anyoptimization/pymoo
- **File**: pymoo/problems/dynamic/df.py
- **Used for**: Cross-validation of formulas AND variable bounds (Level 2)
- **Access date**: 2026-10-06
- **Status**: INDEPENDENT_REFERENCE (confirms MATLAB on all formulas; provides bounds)

### S7: Official MATLAB PF Sampling Code (SUCCESSFULLY ACCESSED)
- **Title**: cec2018_DF_PF.m
- **URL**: http://homepages.cs.ncl.ac.uk/shouyong.jiang/cec2018/cec2018_DF_PF.m
- **Used for**: True PF generation reference
- **Access date**: 2026-10-06
- **Status**: AUTHORITATIVE (successfully downloaded)

## SOURCE CROSS-VALIDATION: MATLAB vs Python Translation (ahcheriet)

Discrepancies found — MATLAB (S2) is authoritative; pymoo (S6) confirms MATLAB.

| Problem | Field | MATLAB (correct) | ahcheriet Python (S5, WRONG) |
|---------|-------|-------------------|------------------------------|
| DF3 | g | (x_i - G - x₁^H)² | ((x_i - G - x₁)^H)² — H applied to wrong term |
| DF4 | g | a*(x₁/c)²/[2:n] (divide by index) | a*x₁²/x_i (divide by variable) |
| DF8 | tmp | sin(4πx₁) | sin((4πx₁)^b) with b=100G² — spurious exponent |
| DF8 | f₂ | uses x(1) = x₁ | uses x[1] = x₂ — wrong variable |
| DF10 | tmp | sin(4π(x₁+x₂)) | sin(2π(x₁+x₂)) — wrong coefficient |
| DF12 | tmp2 | abs(sin(...)), uses BOTH x₁,x₂ | sin(...), only x₁, no abs |
| DF12 | f₃ | sin(0.5π·x₁) | sin(0.5π·x₂) — wrong variable |

## VARIABLE BOUNDS (from pymoo S6, consistent with MATLAB PF code S7)

| Problem | x₁ bounds | x₂ bounds | x₃..xₙ bounds | n_obj |
|---------|-----------|-----------|----------------|-------|
| DF1 | [0, 1] | [0, 1] | [0, 1] | 2 |
| DF2 | [0, 1] | [0, 1] | [0, 1] | 2 |
| DF3 | [0, 1] | [-1, 2] | [-1, 2] | 2 |
| DF4 | [-2, 2] | [-2, 2] | [-2, 2] | 2 |
| DF5 | [0, 1] | [-1, 1] | [-1, 1] | 2 |
| DF6 | [0, 1] | [-1, 1] | [-1, 1] | 2 |
| DF7 | [1, 4] | [0, 1] | [0, 1] | 2 |
| DF8 | [0, 1] | [-1, 1] | [-1, 1] | 2 |
| DF9 | [0, 1] | [-1, 1] | [-1, 1] | 2 |
| DF10 | [0, 1] | [0, 1] | [-1, 1] | 3 |
| DF11 | [0, 1] | [0, 1] | [0, 1] | 3 |
| DF12 | [0, 1] | [0, 1] | [-1, 1] | 3 |
| DF13 | [0, 1] | [0, 1] | [-1, 1] | 3 |
| DF14 | [0, 1] | [0, 1] | [-1, 1] | 3 |

## E4U DEFECTS CONFIRMED

| Defect | E4U behavior | Official behavior |
|--------|-------------|-------------------|
| DF4 | Identical to DF1 (same g, same f1, same f2) | Completely different: g=1+Σ(x_i-a*x_0²/x_i)², f1=g*|x_0-a|^H, f2=g*|x_0-a-b|^H |
| DF12 | Identical to DF10 | Different: g includes sin(floor(k*(2x_0-1))*π/2) term; f2=g*sin(π/2*x_1)*cos(π/2*x_0); f3=g*sin(π/2*x_1) |
| DF13 | Identical to DF10 | Different: uses cos²/sin² structure with p-dependent knee term |
| DF14 | Identical to DF10 | Different: y=0.5+G*(x_0-0.5) with 6π sinusoidal perturbation |
