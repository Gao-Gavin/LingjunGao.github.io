---
title: "Dimension Reduction Project"
excerpt: "Try to using the low dimensional vector to represent the original data and explain the structure behind the data<br/><img src='/images/PCA.png'>" 
collection: portfolio
---

<style>
  body {
      font-size: 14px;  /* 调整字体大小 */
  }
  .post-content, .page-content {
      font-size: 14px;
      line-height: 1.5;
  }
</style>

In the context of the high dimension data, we always want to find some low dimensional vector to represent the original high dimension data.
## Factor Analysis Model

Suppose we need to use the low dimensional vector to represent the
original data, the original data $Y$ is not centralized yet. Such
representation follows the following expression:

$$
Y_i^{*}=\mu+\Lambda X_i+\epsilon_i
$$

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



According this algorithm, I can have the the code shown as follows:

```{r}
FA_generalized=function(Y,K=2,sigma,epsilon = 1e-3,max_iter=200){
    #set.seed(601)
    n=dim(Y)[1]
    p=dim(Y)[2]

    lambda=matrix(rnorm(p*K,0,sigma),nrow=p,ncol=K)
    phi=diag(rnorm(p,0,1))

    loss=c()
    
    original=matrix(c(rep(1,4),numeric(3),numeric(3),rep(1,4)),nrow=7,ncol=2)%*%t(matrix(c(rep(1,4),numeric(3),numeric(3),rep(1,4)),nrow=7,ncol=2))
    new=lambda%*%t(lambda)
    diff=norm(new-original,type="F")
    
    t=0


    while(diff>epsilon){
        t=t+1
    
        Varx=diag(1,K)-t(lambda)%*%solve(lambda%*%t(lambda)+phi)%*%lambda
        Ex=Y%*%solve(lambda%*%t(lambda)+phi)%*%lambda
        Extx=n*Varx+t(Ex)%*%Ex
    
        lambda_new=t(Y)%*%Ex%*%solve(Extx)
        

        lambda=lambda_new
    
        phi_new=diag(diag(1/n*(t(Y)%*%Y-2*t(Y)%*%Ex%*%t(lambda)+lambda%*%Extx%*%t(lambda))))
        phi=phi_new

        Sigma_y=lambda%*%t(lambda)+phi+epsilon * diag(p) 

        loss<-c(loss,(-n*p/2)*log(2*pi)-(n/2)*log(abs(det(Sigma_y)))-0.5*sum(diag(solve(Sigma_y)%*%t(Y)%*%Y)))
        new=lambda%*%t(lambda)
        diff=norm(new-original,type="F")
        if(t>max_iter){
          break
        }
        }
    #print(paste("Times of iteration=",i))
    return(list(lambda=lambda,phi=phi,loss=loss))
    }

```
