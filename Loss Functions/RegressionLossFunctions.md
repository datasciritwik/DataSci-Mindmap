Okay, here's a cheat sheet for regression loss functions, designed for quick review before an interview.

---

# Regression Loss Functions: Interview Cheat Sheet

**Core Idea:** Regression loss functions quantify the difference between the true continuous values $(y)$ and the predicted continuous values $(\hat{y})$. The goal of model training is to minimize this loss.

**Notation:**
*   $N$: Number of samples
*   $y_i$: True value for the $i$-th sample
*   $\hat{y}_i$: Predicted value for the $i$-th sample
*   $e_i = y_i - \hat{y}_i$: Error for the $i$-th sample

---

## 1. Mean Squared Error (MSE) / L2 Loss

*   **Formula:**
    $$ \text{MSE}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 $$
*   **Description:** Calculates the average of the squared differences between true and predicted values.
*   **Pros:**
    *   Smoothly differentiable, making optimization easier (convex function with a unique minimum).
    *   Penalizes larger errors more significantly due to squaring.
*   **Cons:**
    *   Very sensitive to outliers (squaring magnifies large errors).
    *   The units are squared (e.g., if predicting price in USD, MSE is in USD²), making it less interpretable directly.
*   **When to Use:**
    *   General-purpose, common default.
    *   When large errors are particularly undesirable and should be heavily penalized.
    *   When outliers are not a major concern or have been handled.

---

## 2. Root Mean Squared Error (RMSE)

*   **Formula:**
    $$ \text{RMSE}(y, \hat{y}) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2} = \sqrt{\text{MSE}} $$
*   **Description:** The square root of the MSE.
*   **Pros:**
    *   Same units as the target variable, making it more interpretable than MSE.
    *   Still penalizes larger errors more significantly.
    *   Differentiable (gradient is related to MSE's gradient).
*   **Cons:**
    *   Still sensitive to outliers (like MSE).
*   **When to Use:**
    *   When you want the error metric in the same units as the target variable for better interpretation.
    *   Similar situations as MSE, often preferred for reporting error.

---

## 3. Mean Absolute Error (MAE) / L1 Loss

*   **Formula:**
    $$ \text{MAE}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i| $$
*   **Description:** Calculates the average of the absolute differences between true and predicted values.
*   **Pros:**
    *   Robust to outliers (doesn't square errors).
    *   Interpretable, as it's in the same units as the target variable.
*   **Cons:**
    *   Not differentiable at zero (where $y_i = \hat{y}_i$). Sub-gradients can be used.
    *   May lead to multiple solutions (less stable minima compared to MSE).
    *   Doesn't penalize large errors as heavily as MSE.
*   **When to Use:**
    *   When the dataset contains outliers that you don't want to dominate the loss.
    *   When all errors, regardless of magnitude, should be treated proportionally.

---

## 4. Huber Loss (Smooth Mean Absolute Error)

*   **Formula:**
    $$ L_{\delta}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} \begin{cases} \frac{1}{2}(y_i - \hat{y}_i)^2 & \text{for } |y_i - \hat{y}_i| \le \delta \\ \delta (|y_i - \hat{y}_i| - \frac{1}{2}\delta) & \text{for } |y_i - \hat{y}_i| > \delta \end{cases} $$
    where $\delta$ is a hyperparameter.
*   **Description:** A combination of MSE and MAE. Quadratic for small errors (like MSE) and linear for large errors (like MAE).
*   **Pros:**
    *   Less sensitive to outliers than MSE (due to linear component for large errors).
    *   Differentiable everywhere.
    *   Combines the best of both MSE (stable minimum for small errors) and MAE (robustness).
*   **Cons:**
    *   Requires tuning the hyperparameter $\delta$, which defines the transition point.
*   **When to Use:**
    *   When you want a balance between MSE's sensitivity to large errors and MAE's robustness.
    *   Good compromise when dealing with potential outliers.

---

## 5. Log-Cosh Loss

*   **Formula:**
    $$ L(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} \log(\cosh(\hat{y}_i - y_i)) $$
*   **Description:** A smooth function that is similar to MSE for small errors and to log(abs(error)) (similar to MAE) for large errors. $\cosh(x) = \frac{e^x + e^{-x}}{2}$.
*   **Pros:**
    *   Twice differentiable everywhere (smoother than Huber Loss).
    *   Generally robust to outliers.
*   **Cons:**
    *   Computationally more expensive than MSE or MAE.
*   **When to Use:**
    *   As a smoother alternative to Huber loss.
    *   When a high degree of smoothness is desired along with robustness.

---

## 6. Quantile Loss (Pinball Loss)

*   **Formula (for a single prediction and target, for quantile $\tau$):**
    $$ L_{\tau}(y, \hat{y}) = \begin{cases} \tau (y - \hat{y}) & \text{if } y - \hat{y} > 0 \\ (1 - \tau) (\hat{y} - y) & \text{if } y - \hat{y} \le 0 \end{cases} $$
    Averaged over all $N$ samples: $\frac{1}{N} \sum L_{\tau}(y_i, \hat{y}_i)$.
*   **Description:** Used for quantile regression. It asymmetrically penalizes over-predictions and under-predictions based on the quantile $\tau \in (0,1)$.
    *   If $\tau = 0.5$, it's equivalent to MAE (predicting the median).
    *   If $\tau > 0.5$, it penalizes under-prediction more.
    *   If $\tau < 0.5$, it penalizes over-prediction more.
*   **Pros:**
    *   Allows prediction of specific quantiles, not just the mean. Useful for understanding uncertainty / prediction intervals.
    *   Robust to outliers (especially for $\tau=0.5$).
*   **Cons:**
    *   Requires choosing the quantile $\tau$.
    *   Not differentiable at $y = \hat{y}$.
*   **When to Use:**
    *   When you need to predict a specific quantile of the target distribution (e.g., 10th, 50th, 90th percentile).
    *   To create prediction intervals.

---

## 7. Mean Absolute Percentage Error (MAPE)

*   **Formula:**
    $$ \text{MAPE}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100\% $$
*   **Description:** Calculates the average of absolute percentage errors.
*   **Pros:**
    *   Scale-independent, expresses error as a percentage, which is often intuitive for business stakeholders.
*   **Cons:**
    *   **Undefined** if any true value $y_i = 0$.
    *   Can lead to extremely large values if $y_i$ is close to zero.
    *   Asymmetric: Puts a heavier penalty on negative errors (when $\hat{y}_i > y_i$) than on positive errors if $y_i > 0$. For example, if $y_i=100$, an error of $+10$ ($\hat{y}_i=110$) is $10\%$. An error of $-10$ ($\hat{y}_i=90$) is also $10\%$. But if $y_i=10$, an error of $+90$ ($\hat{y}_i=100$) is $900\%$, while an error of $-10$ ($\hat{y}_i=0$) is $100\%$. It particularly penalizes predictions that are too low more than predictions that are too high by the same absolute amount, *relative to the true value*. (More accurately, it has no upper bound for positive errors when $y_i > 0$, but is bounded by 100% for negative errors that result in $\hat{y}_i \ge 0$).
*   **When to Use:**
    *   When relative error is important and interpretability as a percentage is desired.
    *   Ensure $y_i$ is never zero or very close to zero. Common in demand forecasting.

---

## 8. Symmetric Mean Absolute Percentage Error (SMAPE)

*   **Formula (common definition, varies):**
    $$ \text{SMAPE}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2} \times 100\% $$
*   **Description:** An attempt to address MAPE's asymmetry by normalizing with the average of the absolute true and predicted values.
*   **Pros:**
    *   Bounds the error between 0% and 200% (for this definition).
    *   More symmetric than MAPE in how it treats over- and under-predictions.
*   **Cons:**
    *   Still problematic if both $y_i$ and $\hat{y}_i$ are zero or very close to zero (denominator becomes zero).
    *   Definition can vary, leading to confusion.
    *   Not truly symmetric if one value is zero and the other isn't.
    *   Interpretation can be less straightforward than MAPE.
*   **When to Use:**
    *   When a percentage error metric is needed, and MAPE's asymmetry or unboundedness is problematic.
    *   Be cautious about the specific definition used.

---

## Key Takeaways for Interviews:

*   **No "best" loss function:** Choice depends on the problem, data characteristics (esp. outliers), and what aspects of error are most critical.
*   **MSE/RMSE:** Good defaults, penalize large errors, but sensitive to outliers. RMSE is more interpretable.
*   **MAE:** Robust to outliers, but gradient issues at zero.
*   **Huber/Log-Cosh:** Balance between MSE and MAE, offering robustness with good mathematical properties. Huber needs $\delta$. Log-Cosh is smoother.
*   **Quantile Loss:** For predicting ranges/uncertainty, not just point estimates.
*   **MAPE/SMAPE:** For percentage errors, but beware of $y_i=0$ issues and (S)MAPE's specific quirks.

Be ready to discuss the trade-offs (sensitivity to outliers, differentiability, interpretability) of each! Good luck!