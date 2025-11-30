# Curvature-Adaptive-RMS-Momentum-Acceleration-Optimiser
Curvature Adaptive RMS Momentum Acceleration Optimiser

CARMA is an optimization algorithm designed to combine the strengths of Nesterov's accelerated gradient, curvature-adaptive momentum, and RMS scaling.
It provides fast convergence in smooth regions, stability in sharp or curved regions.

Introduction
Modern optimization methods face a trade-off where they have to choose between traversal of flat landscapes and stable behaviour in high-curvature regions. Optimizers such as Nesterov Accelerated Gradient improve convergence speed by utilising look-ahead updates, but their momentum coefficient is usually static. RMSProp stabilizes training by normalizing the gradients using a running estimate of their second moment. However, it also lacks curvature awareness and might oscillate in regions where the gradient direction changes rapidly.
To address these limitations, CARMA introduces a new mechanism that regulates momentum using curvature feedback - derived from differences between consecutive gradients. This allows our optimizer to slow down before steep cliffs and speed up when entering broad valleys.

Key Innovations

1. Curvature-Aware Momentum
Based on the gradient curvature, our optimizer adapts the momentum coefficient β. Momentum decreases in high curvature regions to prevent overshooting, and increases in flatter regions to accelerate progress.
2. RMS-Stabilized Velocity Updates
Instead of moment estimates like Adam, CARMA normalizes the momentum using an RMS running average of squared gradients, which provides stable, scaled updates.
3. Soft Momentum Reset
When the gradient direction changes rapidly, CARMA automatically reduces the effective momentum.This acts like a soft reset mechanism that helps in avoiding oscillation around valleys and saddle points.

Inspiration and Development

The idea of CARMA emerged from observing Nesterov Accelerated Gradient (NAG). Despite the lookahead mechanism, NAG still struggles in regions where the curvature changes rapidly. The fixed momentum in NAG, helps accelerate movement across flat regions, but in steep or highly curved sections it can lead to overshooting and instability. Additionally, NAG applies the same update scale across all coordinates, which makes it sensitive to variations in gradient magnitude.Incorporating adaptive momentum based on curvature helps moderate the update speed in steep regions while maintaining acceleration in flat areas. And RMS Scaling would normalize high-variance direction, smoothening noisy gradients. Initial experiments on simple quadratic functions validated this hypothesis, leading to the development of the full algorithm.

Mathematical Updates: 
CARMA Algorithm: 
    yt = xt-1​ − ηβ0​vt-1    // Nesterov lookahead
    gₜ = ∇f(yₜ)    // gradient
    // estimate curvature
    if t > 1:
        κₜ = ||gₜ - gₜ₋₁|| / (|| gₜ₋₁|| + ε )
    else:
        κₜ = 0
    // Adaptive momentum coefficient
    βₜ = β0 / (1 + αc·κₜ )
    for the momentum bounds:  βₜ = min(βmax , max(βmin ,  βₜ))
    st = ρs . st-1 ​+(1−ρ)gt2   // RMS Scaling
    vₜ = βt·vₜ₋₁ + (1 - βt)·gₜ   // Momentum update
    xₜ = xₜ₋₁ - η · vₜ / (√sₜ + ε)     // Parameter update

  Notations
  
xt : parameters at iteration t
yt : Nesterov lookahead point
gt : gradient at the look-ahead
vt : momentum
st : RMS Term
κₜ : Curvature signal

Algorithm Code: 
# CARMA
def carma(grad_func, x0, lr=0.12, beta0=0.95, alpha_c=0.3,beta_min=0.9, beta_max=0.99, rho_s=0.99, eps=1e-8,max_iter=5000, tol=1e-6, grad_clip=50.0, args=()):
    x = x0.copy()
    v = np.zeros_like(x)
    s = np.zeros_like(x)
    g_prev = np.zeros_like(x)
    losses = []
    start = time.time()
    for i in range(max_iter):
        y = x - lr * beta0 * v  //lookahead 
        g = grad_func(y, *args)  //gradient
        g = np.clip(g, -grad_clip, grad_clip)
        c = np.linalg.norm(g - g_prev) / (np.linalg.norm(g_prev) + eps)  //estimating curvature
        beta_t = beta0 / (1 + alpha_c * c)   //adaptive momentum
        beta_t = np.clip(beta_t, beta_min, beta_max)
        s = rho_s * s + (1 - rho_s) * (g * g)   //RMS update
        v = beta_t * v + (1 - beta_t) * g  //momentum update
        x = x - lr * v / (np.sqrt(s) + eps)  //parameter update
        losses.append(np.linalg.norm(g))
        g_prev = g
        if losses[-1] < tol:
                break
    return x, i+1, losses, time.time() - start

    
How does this algorithm work?
1. In the loop, first it computes a look-ahead position before taking the gradient.
2. Compute the gradient at the lookahead point
3. Clip it to prevent exploding gradients.
4. The curvature estimation, c, measures how much gradient changed direction. If the difference is large, it leads to high curvature. If the difference is small, it leads to low curvature. This is the key for adaptive momentum.
5. Adaptive momentum, beta_t, decreases when the curvature is high.
   As c is large, denominator becomes big, beta_t becomes small, leading to slower momentum. This ensures stability inside sharp valleys.
6. s, is the RMS vector, which builts the exponential average of squared gradients. This normalises the update step size and smooths noisy gradients.
7. This is the momentum update, same as Adam's first moment, but with adaptive momentum. This helps keep direction and allows controlled acceleration.
8. Updates go in momentum direction but shrink when gradient magnitude is large.
9. losses store the gradient norm, to monitor the convergence, and the gradient is saved.
10. It stops when the gradient becomes extremely small.


Default hyperparameter values:
- η (learning rate): 0.12
- β0 (base momentum): 0.95
- βmin (minimum momentum): 0.9
- βmax (maximum momentum): 0.99
- (numerical stability): 1e-8
- ρ (RMS decay rate): 0.99
- αc (curvature sensitivity): 0.3
- gradient clipping threshold: 50.0
