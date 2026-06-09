# Treatment Effect Estimation

A multi-stage model for individual treatment effect (ITE) estimation. A shared encoder splits
the input into a **stable** representation $S$ and a **confounding** representation $C$;
downstream heads model propensity, outcomes, doubly-robust ITE targets, and a conditional
diffusion model over $C$. Stages are trained sequentially with Adam, early stopping, and
`reduction="sum"` MSE/CE losses (see `train_all` in `train.py`).

## Shared notation

| Symbol            | Code      | Meaning                                   |
| ---               | ---       | ---                                       |
| $x$               | `x`       | Input covariates                          |
| $S,\ C$           | `S`, `C`  | Stable / confounding representation       |
| $t,\ y$           | `t`, `y`  | Treatment ($K$ groups) / observed outcome |
| $\Phi_\phi$       | `enc`     | Encoder, maps $x$ into $S, C$   |
| $\pi_\beta(C)$    | `prop`    | Propensity head                           |
| $f_\theta(S,C,t)$ | `out`     | Outcome model                             |
| $g_\omega(S, C)$  | `ite`     | Auxiliary ITE head                        |
| $-$               | `denoiser`| Denoiser/Diffusion Model                  | 

| Setting            | Constant                         | Default              |
| ---                | ---                              | ---                  |
| LR / weight decay  | `LEARNING_RATE` / `WEIGHT_DECAY` | `1e-3` / `1e-5`      |
| Epochs / Batch     | `EPOCHS` / `BATCH_SIZE`          | `100` / `32`         |
| Patience           | `PATIENCE`                       | `15`                 |
| Seed / splits      | `SEED` / `TRAIN,VAL,TEST_SPLIT`  | `42` / `0.7,0.2,0.1` |

## Training Objectives (Full Loss Function)

### 1. Representation Learning (`train_representation`)

Augment each batch into two views $x^{(1)}, x^{(2)}$, encode to $(S^{(i)}, C^{(i)})$. The 
loss function is a **weighted combination** of four components:


```math
\mathcal{L}_{\text{rep}}
= \lambda_{\text{stab}}\,\mathcal{L}_{\text{stab}}
+ \lambda_{\text{prop}}\,\mathcal{L}_{\text{prop}}
+ \lambda_{\text{mmd}}\,\mathcal{L}_{\text{mmd}}
+ \lambda_{\text{ac}}\,\bigl(\mathcal{L}_{\text{ac}}^{S} + \mathcal{L}_{\text{ac}}^{C}\bigr)
```


| Component                    | Definition                                                        | Purpose                                                         |
| ---                          | ---                                                               | ---                                                             |
| $\mathcal{L}_{\text{stab}}$  | $\mathrm{MSE}\bigl(S^{(1)}, S^{(2)}\bigr)$                        | $S$ invariant across views                                      |
| $\mathcal{L}_{\text{prop}}$  | $\tfrac{1}{2}\sum_i \mathrm{CE}\bigl(\pi_\beta(C^{(i)}), t\bigr)$ | $C$ predicts treatment                                          |
| $\mathcal{L}_{\text{mmd}}$   | $\tfrac{1}{2}\sum_i \mathrm{MMD}\bigl(C^{(i)}, t\bigr)$           | $C$ balanced across treatments (RBF-kernel $MMD^2$)             |
| $\mathcal{L}_{\text{ac}}(z)$ | $-$                                                               | Anti-collapse regularization (decorrelation & variance control) |

| Hyperparameter                                                       | Constant               | Default |
| ---                                                                  | ---                    | ---     |
| $\lambda_{\text{stab}}, \lambda_{\text{mmd}}, \lambda_{\text{prop}}$ | `LAMBDA_STAB/MMD/PROP` | `1.0`   |
| $\lambda_{\text{ac}}$                                                | `LAMBDA_AC`            | `0.1`   |

### 2. Outcome Model (`train_outcome`)
Single-term regression loss between the factual prediction $(f_\theta(S, C, t))$ and the observed outcome $(y)$.

$$
\mathcal{L}_{\text{out}} = \mathrm{MSE}\bigl(f_\theta(S, C, t),\, y\bigr)
$$


### 3. Nuisances (`train_nuisances`)

Cross-fitted (folds A/B): PropensityHead + OutcomeModel for the doubly-robust targets:

$$
\mathcal{L}_{\text{nuis}} = \mathrm{CE}\bigl(\pi_\beta(C), t\bigr) + \mathrm{MSE}\bigl(f_\theta(S, C, t), y\bigr)
$$


### 4. ITE head (`train_ite`)

Provides Individual Treatment Effect (ITE) relative to the most frequent treatment in training.

$$
\mathcal{L}_{\text{ite}} = \mathrm{MSE}\bigl(g_\omega(S, C),\, \tau\bigr)
$$

where $\tau$ (`effect_target`) is the doubly-robust target from the cross-fitted nuisances.

| Setting                | Constant                  | Default         |
| ---                    | ---                       | ---             |
| Propensity clip / trim | `CLIPPING_EPS` / `TRIM`   | `0.05` / `0.05` |
| Winsorize quantiles    | `WINSORIZATION_QUANTILES` | `(0.01, 0.99)`  |

### 5. Diffusion denoiser (`train_denoiser`)

DDPM $\varepsilon$-prediction over frozen $C$, conditioned on step $s$, $S$, and $t$:

$$
\mathcal{L}_{\text{diff}} = \mathrm{MSE}\bigl(\hat{\varepsilon_\theta}, \varepsilon\bigr)
$$