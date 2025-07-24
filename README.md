# 🧠 Gradient Descent Optimizers Comparison

This document summarizes the key differences between popular optimization algorithms used in machine learning and deep learning. Each optimizer improves over the basic gradient descent in a specific way.

---

## 🟩 1. Batch Gradient Descent (BGD)
- Used mainly for **small datasets**.
- Updates weights using **all training data in each iteration**.
- Very **stable** but **slow**.
- Performs one full pass over the data (epoch) each time.
- Can get stuck in **saddle points** or **local minima**.

---

## 🟦 2. Stochastic Gradient Descent (SGD)
- Uses **one random sample** to update the parameters at each step.
- Much **faster** than BGD.
- Helps avoid **saddle points** thanks to its **noisy nature**.
- However, convergence can be unstable due to **high variance** in updates.

---

## 🟨 3. Mini-Batch Gradient Descent
- Combines BGD and SGD by working on **small batches** of data (e.g. 32 or 64 samples).
- Helps speed up training and smooths out the noise.
- Commonly used in **deep learning**.
- May still face issues like **vanishing gradients** in some cases.

---

## 🟧 4. Momentum
- Speeds up convergence by adding **
- v_t = γ * v_{t-1} + α * ∇J(θ_t)
- θ_{t+1} = θ_t - v_t

velocity** based on past gradients.
- Uses the direction of previous steps to avoid local minima and damp oscillations.
- Equations:

- v_t = γ * v_{t-1} + α * ∇J(θ)
- θ_{t+1} = θ_t - v_t

- Drawback: It may overshoot or continue in the wrong direction if not corrected.
---
## 🟥 5. Nesterov Accelerated Gradient (NAG)
- Improves Momentum by **looking ahead** before computing the gradient.
- Helps reduce overshooting by being more informed about future direction.
- Equations:
- θ_temp = θ_t - γ * v_{t-1}
- v_t = γ * v_{t-1} + α * ∇J(θ_temp)
- θ_{t+1} = θ_temp - α * ∇J(θ_temp)
---
## 🟪 6. Adagrad
- Great for **sparse data** and features with different frequencies.
- Uses an **adaptive learning rate per parameter**.
- Learning rate decreases over time (can become too small).
- Equations:
- v_t = v_{t-1} + (∇J(θ))²
- θ_{t+1} = θ_t - α / (√v_t + ε) * ∇J(θ)

- `ε` prevents division by zero.

---

## 🟫 7. RMSProp
- Extension of Adagrad using **exponential moving average** to control learning rate.
- Prevents the learning rate from shrinking too much.
- Equations:
- v_t = β * v_{t-1} + (1 - β) * (∇J(θ))²
- θ_{t+1} = θ_t - α / (√v_t + ε) * ∇J(θ)


---

## 🟩 8. Adam (Adaptive Moment Estimation)
- Combines **Momentum** and **RMSProp**.
- Keeps track of both:
- First moment (mean of gradients): `m_t`
- Second moment (uncentered variance): `v_t`
- Uses **bias correction** to stabilize early updates.
- Equations:
- m_t = β₁ * m_{t-1} + (1 - β₁) * ∇J(θ)
- v_t = β₂ * v_{t-1} + (1 - β₂) * (∇J(θ))²

- m̂_t = m_t / (1 - β₁^t)
- v̂_t = v_t / (1 - β₂^t)

- θ_{t+1} = θ_t - α / (√v̂_t + ε) * m̂_t


---

## 🧩 Why Do We Use Bias Correction in Adam?

In the early iterations of training, both `m_t` and `v_t` are initialized as **zeros**. This means:

- Their initial values are **biased toward zero**, especially in the first few steps.
- Without correction, the optimizer would **underestimate** the actual gradient and variance.

To fix that, we apply **bias correction**:

- `m̂_t = m_t / (1 - β₁^t)`
- `v̂_t = v_t / (1 - β₂^t)`

This ensures:
- The moments are **unbiased estimators**, especially in early stages.
- **Stable and accurate updates** from the beginning of training.

---

## sympols
- `α`: Learning rate  
- `γ`: Momentum factor  
- `β₁`, `β₂`: Decay rates (Adam)  
- `∇J(θ)`: Gradient of the cost function  
- `v_t`: Accumulated squared gradients (variance)  
- `m_t`: Accumulated gradients (mean)  
- `ε`: Small constant to avoid divide-by-zero  
---
