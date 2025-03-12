---
title: "EM alogorithm in Factor Analysis"
excerpt: "Short description of portfolio item number 1<br/><img src='/images/500x300.png'>" 
collection: portfolio
---

## Factor Analysis Model

Suppose we need to use the low dimensional vector to represent the
original data, the original data $Y$ is not centralized yet. Such
representation follows the following expression:

$$
Y_i^{*}=\mu+\Lambda X_i+\epsilon_i
$$

And we have the following assumption as follows:

-   $
    X_i\sim N_k(0,I_k) 
    $

-   $
    \epsilon_i\sim N_p(0,\Phi) \ \ \ \text{where } \Phi=diag[\Phi_1,\Phi_2,\ldots,\Phi_p]
    $

-   $
    X_i\perp \epsilon_j \ \text{for any } x,y.
    $

And we can have the joint distribution as follows:
$$
\begin{bmatrix}
X\\
Y^{*}
\end{bmatrix}
\sim N_{p+k}(
\begin{bmatrix}
\mathbb{0}\\
\mu
\end{bmatrix},\begin{bmatrix}
I&\Lambda^T\\
\Lambda&\Lambda\Lambda^T+\Phi\\
\end{bmatrix})
$$

According to the joint distribution, we can get the marginal
distribution and conditional distribution as follows:

-   $
    Y_i^{*}\sim N_p(\mu,\Lambda\Lambda^T+\Phi)
    $

-   $
    X_i\mid Y_i^{*}\sim N_k(\Lambda^T(\Lambda\Lambda^T+\Phi)^{-1}(Y_i^{*}-\mu),I-\Lambda^T(\Lambda\Lambda^T+\Phi)^{-1}\Lambda)
    $

We can know the Expectation about the distribution of $X$ conditional on
$Y$ as follows:

$$
\begin{aligned}
\begin{cases}
E(X\mid Y, \Lambda,\Phi)=(Y-\mu)(\Lambda\Lambda^T+\Phi)^{-1}\Lambda\\
E(X^TX)=\sum_{i}^{N}\{(Var(X_i)+E(X_i)E(X_i)^T\}=n(I-\Lambda^T(\Lambda\Lambda^{-1}+\Phi)^{-1}\Lambda)-E(X\mid Y, \Lambda,\Phi)^TE(X\mid Y, \Lambda,\Phi)
\end{cases}
\end{aligned}
$$

### Estimate the Sample Mean by MLE

$$
f_{Y_i^{*}}(y_i^{*})=(2\pi)^{-\frac{p}{2}}|\Lambda\Lambda^T+\Phi|^{-\frac{1}{2}}\exp{\{-\frac{1}{2}(y_i^{*}-\mu)^T(\Lambda\Lambda^T+\Phi)^{-1}(y_i^{*}-\mu)\}}
$$

Hence, the sample distribution is

$$
f_{Y^{*}}(y^{*})=\prod_{i=1}^{N}f_{Y_i^{*}}(y_i^{*})=(2\pi)^{-\frac{Np}{2}}|\Lambda\Lambda^T+\Phi|^{-\frac{N}{2}}\exp{\{-\frac{1}{2}\sum_{i=1}^{N}(y_i^{*}-\mu)^T(\Lambda\Lambda^T+\Phi)^{-1}(y_i^{*}-\mu)\}}
$$

We can take the log operation to simplify the calculation and maximize
the Log-Likelihood function. The Log-Likelihood function can be
expressed as follows:

$$
\begin{aligned} 
l(\theta) &= \ln f_{Y^{*}}(y^{*})\\
&= -\frac{Np}{2} \ln (2\pi) 
   - \frac{N}{2} \ln \left| \Lambda \Lambda^T + \Phi \right|
   - \frac{1}{2} \sum_{i=1}^{N} (y_i^{*} - \mu)^T (\Lambda \Lambda^T + \Phi)^{-1} (y_i^{*} - \mu)\\\\
&= -\frac{Np}{2} \ln (2\pi) 
   - \frac{N}{2} \ln \left| \Lambda \Lambda^T + \Phi \right|
   - \frac{1}{2} \sum_{i=1}^{N} \left[ {y_i^{*}}^T (\Lambda \Lambda^T + \Phi)^{-1} y_i^{*} 
   - 2\mu^T (\Lambda \Lambda^T + \Phi)^{-1} y_i^{*} 
   + \mu^T (\Lambda \Lambda^T + \Phi)^{-1} \mu \right]
\end{aligned}
$$

According to the MLE method we can derive the as follow:

$$
\frac{\partial}{\partial \mu} l(\mu) = 0 
\Rightarrow
-\sum_{i=1}^{N} (\Lambda \Lambda^T + \Phi)^{-1} y_i^{*}
+ N (\Lambda \Lambda^T + \Phi)^{-1} \mu = 0
\Rightarrow
\hat{\mu} = \frac{\sum_{i=1}^{N} y_i^{*}}{N} = \bar{y^{*}}
$$

In this case we can centralize the data as the following expression:

$$
Y\leftarrow Y^{*}-\bar{Y^{*}}
$$

## EM algorithm to solve the $\Lambda$ and $\Phi$

However, we can not solve the $\Lambda$ directly since $\Lambda$ is not
identifiable. So we may use the EM algorithm to solve this problem.

$$
P(Y\mid X, \Lambda, \Phi)=\frac{P(X\mid Y, \Phi, \Lambda)P(Y\mid \Lambda, \Phi)}{P(X\mid \Lambda, \, \Phi)} \Rightarrow P(Y\mid \Lambda, \Phi)=\frac{P(Y\mid X, \Lambda, \Phi)P(X)}{P(X\mid Y, \Phi, \Lambda)}
$$

We can do a log-transformation to make the computation easier:

$$
\begin{aligned}
l(\theta)&=\ln P(Y\mid \Lambda, \Phi)\\
&=\ln P(Y\mid X, \Lambda, \Phi)+ \ln \frac{P(X)}{P(X\mid Y, \Phi, \Lambda)} \ \ \ \ \text{take the expectation in the two sides.}\\
&=\int \ln P(Y\mid X, \Lambda, \Phi) P(X) dX+\int \ln \frac{P(X)}{P(X\mid Y, \Phi, \Lambda)} P(X)dX\\
&=E_{X\sim P(X)}\{\ln P(Y\mid X, \Lambda, \Phi) \}+\mathbb{KL}(P(X)\mid P(X\mid Y, \Phi, \Lambda))
\end{aligned}
$$

Since, KL divergence my always be positive so we can have the following
inequality:

$$
l(\theta)\geq E_{X\sim P(X)}\{\ln P(Y\mid X, \Lambda, \Phi) \}
$$

### E-step

To make the equality hold. The KL divergence must be zero and it
indicates that

$$
\mathbb{KL}(P(X)\mid P(X\mid Y, \Phi, \Lambda))=0 \Rightarrow P(X) \overset{d}{=}P(X\mid Y, \Phi, \Lambda)
$$

Hence, the likelihood function transfer into a expectation as follows:

$$
l(\theta)=E_{X\sim P(X\mid Y,\Lambda, \Phi)}\{P(Y\mid X, \Lambda, \Phi) \}
$$

### M-step

To update the $\lambda$ and $\Phi$, we can maximize the $l(\theta)$ and
update the parameters as follows:

$$
(\Lambda^{(t+1)},\Phi^{(t+1)})=\arg\max_{\Lambda,\Phi}E_{X\sim  P(X\mid Y,\Lambda^{(t)}, \Phi^{(t)})}\{\ln P(Y\mid X, \Lambda, \Phi) \}
$$

We can know the distribution of $Y$ given $X$ as follows:

$$
\begin{aligned}
\ln f_{Y\mid X}(y)&=-\frac{Np}{2}\ln(2\pi)-\frac{N}{2}\ln|\Phi|-\text{Trace}\{\frac{1}{2}\sum_{i=1}^{N}(y_i-\Lambda x_i)^T\Phi^{-1}(y_i-\Lambda x_i)\}\\
&=-\frac{Np}{2}\ln(2\pi)-\frac{N}{2}\ln|\Phi|-\frac{1}{2}\text{Trace}\{\Phi^{-1}\sum_{i=1}^{N}(y_i-\Lambda x_i)(y_i-\Lambda x_i)^T\}\\
&=-\frac{Np}{2}\ln(2\pi)-\frac{N}{2}\ln|\Phi|-\frac{N}{2}\text{Trace}\{\Phi^{-1}S_y\} \ \ \ \ \text{where }S_y=\frac{1}{N}\sum_{i=1}^{N}(y_i-\Lambda x_i)(y_i-\Lambda x_i)^T\\
\Rightarrow\\
E_{X\sim P(X\mid Y,\Lambda, \Phi)}\{P(Y\mid X, \Lambda, \Phi) \}&=-\frac{Np}{2}\ln(2\pi)-\frac{N}{2}\ln|\Phi|-\frac{N}{2}\text{Trace}\{\Phi^{-1}E(S_y)\}
\end{aligned}
$$

We need to solve the $E(S_y)$, the the detail derivation is expressed as
follows:

$$
\begin{aligned}
E(S_y)&=\frac{1}{N}E\{\sum_{i=1}^{N}(y_i-\Lambda x_i)(y_i^T-x_i^T\Lambda^T )\}\\
&=\frac{1}{N}E\{\sum_{i=1}^{N}y_iy_i^T-y_ix_i^T\Lambda^T -\Lambda x_iy_i^T-\Lambda x_ix_i^T\Lambda^T\}\\
&=\frac{1}{N}E\{Y^TY-2\Lambda X^TY+\Lambda X^TX\Lambda^T\}\\
&=\frac{1}{N}[Y^TY-2\Lambda E(X)^T Y+\Lambda E(X^TX)\Lambda^T]
\end{aligned}
$$

Take the differentiate with respective to $\Lambda$, we can get the
following result:

$$
\begin{aligned}
\frac{\partial}{\partial \Lambda}E_{X\sim P(X\mid Y,\Lambda, \Phi)}\{P(Y\mid X, \Lambda, \Phi) \}&=-\frac{N}{2}\text{Trace}\{\frac{1}{N}[\Phi^{-1}Y^TY-2\Phi^{-1}\Lambda E(X)^T Y+\Phi^{-1}\Lambda E(X^TX)\Lambda^T]\}\\
&=-\frac{N}{2}[-\Phi^{-1}Y^TE(X)+2\Phi^{-1}\Lambda E(X^TX)]\\
&=0\\
\Rightarrow \hat{\Lambda}^{(t+1)}&=Y^TE(x)[E(X^TX)]^{-1}\\
\end{aligned}
$$

Take the differentiate with respective to $\Lambda$, we can get the
following result:

$$
\begin{aligned}
\frac{\partial}{\partial \Phi}E_{X\sim P(X\mid Y,\Lambda, \Phi)}\{P(Y\mid X, \Lambda, \Phi) \}&=\frac{N}{2}\Phi-\frac{N}{2}E(S)=0 \Rightarrow \hat{\Phi}=E(S)
\end{aligned}
$$

We can calculate the $E(S)$ as follows:

$$
\begin{aligned}
E(S)&=\frac{1}{N}E\{(Y^T-\hat{\Lambda}^{(t+1)} X^T)(Y^T-\hat{\Lambda}^{(t+1)} X^T)^T\}\\
&=\frac{1}{N}E\{(Y^T-\hat{\Lambda}^{(t+1)} X^T)(Y-X{\hat{\Lambda}^{(t+1)}}^T)\}\\
&=\frac{1}{N}E\{(Y^T-\hat{\Lambda}^{(t+1)} X^T)(Y-X{\hat{\Lambda}^{(t+1)}}^T)\}\\
&=\frac{1}{N}E\{Y^TY-2\hat{\Lambda}^{(t+1)}X^T Y+\hat{\Lambda}^{(t+1)} X^TX{\hat{\Lambda}^{(t+1)}}^T\}\\
&=\frac{1}{N}[Y^TY-2\hat{\Lambda}^{(t+1)}E(X)^TY+hat{\Lambda}^{(t+1)} E(X^TX){\hat{\Lambda}^{(t+1)}}^T]
\end{aligned}
$$

Hence, we can know the updated parameters as follows:

$$
\begin{aligned}
&\begin{cases}
\hat{\Lambda}^{(t+1)}=Y^TE(x)[E(X^TX)]^{-1}\\
\hat{\Phi}^{(t+1)}=\frac{1}{N}[Y^TY-2\hat{\Lambda}^{(t+1)}E(X)^TY+hat{\Lambda}^{(t+1)} E(X^TX){\hat{\Lambda}^{(t+1)}}^T]
\end{cases}\\
&\text{Since the following conclusion:}\\
\\
&\begin{cases}
E(X\mid Y, \Lambda,\Phi)=Y(\Lambda\Lambda^T+\Phi)^{-1}\Lambda\\
E(X^TX)=\sum_{i}^{N}\{(Var(X_i)+E(X_i)E(X_i)^T\}=n(I-\Lambda^T(\Lambda\Lambda^{-1}+\Phi)^{-1}\Lambda)-E(X\mid Y, \Lambda,\Phi)^TE(X\mid Y, \Lambda,\Phi)
\end{cases}\\
\Rightarrow\\
&\begin{cases}
\hat{\Lambda}^{(t+1)}=Y^TY(\Lambda^{(t)}{\Lambda^{(t)}}^T+\Phi^{(t)})^{-1}\Lambda^{(t)}[n(I-{\Lambda^{(t)}}^T(\Lambda^{(t)}{\Lambda^{(t)}}^{-1}+\Phi^{(t)})^{-1}\Lambda^{(t)})-{\Lambda^{(t)}}^T(\Lambda^{(t)}{\Lambda^{(t)}}^T+\Phi^{(t)})^{-1}Y^TY(\Lambda^{(t)}{\Lambda^{(t)}}^T+\Phi^{(t)})^{-1}\Lambda^{(t)}]^{-1}\\
\hat{\Phi}^{(t+1)}=\frac{1}{N}\{Y^TY-2\hat{\Lambda}^{(t+1)}{\Lambda^{(t)}}^T(\Lambda^{(t)}{\Lambda^{(t)}}^T+\Phi^{(t)})^{-1}Y^TY+\hat{\Lambda}^{(t+1)} [n(I-{\Lambda^{(t)}}^T(\Lambda^{(t)}{\Lambda^{(t)}}^{-1}+\Phi^{(t)})^{-1}\Lambda^{(t)})-{\Lambda^{(t)}}^T(\Lambda^{(t)}{\Lambda^{(t)}}^T+\Phi^{(t)})^{-1}Y^TY(\Lambda^{(t)}{\Lambda^{(t)}}^T+\Phi^{(t)})^{-1}\Lambda^{(t)}]{\hat{\Lambda}^{(t+1)}}^T\}
\end{cases}\\
\end{aligned}
$$

According this algorithm, I can have the the code shown as follows:

```{r}


```
