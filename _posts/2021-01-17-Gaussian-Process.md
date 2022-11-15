---
title: Gaussian Process
author: euphoria0-0
date: 2021-01-17 22:40:00 +0800
categories: [AI, Machine Learning]
tags: [Machine Learning, PRML, Bayesian, Gaussian Process]
toc: true
math: true
comments: true

---



> *먼저, 이 글은 (대부분) PRML (CH6) 및 GPML (CH2)공부를 바탕으로 작성된 글입니다. 따라서, 간혹 틀리다거나 더 좋은 해석이 있다면 편하게 댓글 부탁드립니다!*



이 글에서는 Gaussian Process와 Gaussian Process를 Regression과 Classification에 적용하는 Gaussian Process Regression, Gaussian Process Classifier에 대해 다룹니다.



# 1. Gaussian Process

------

Gaussian process $$f$$는 수많은 random variable의 컬렉션 중의 어떤 유한한 부분집합이 (multivariate) Gaussian distribution을 따르는 것을 말합니다. 다시 말해, Gaussian random vector를 infinite-dim의 function 형태로 일반화한 것을 이야기 합니다.

GP는 weight가 아닌 function에 대해 prior를 직접 정의합니다. 그럼으로써 infinite function space에서의 distribution을 고려하는 것은 어렵지만, 실제로 input data point(random variable)에 대한 discrete set에서의 function value만 고려하므로 실제로는 finite space에서 생각할 수 있습니다.

GP의 예로, kernel regression을 생각해볼 수 있습니다.


$$
\begin{aligned}
f(\mathbf{x})&=\mathbf{w}^T\phi(\mathbf{x}) \\
p(\mathbf{w})&=\mathcal{N}(\mathbf{w}|\mathbf{0},\alpha^{-1}\mathbf{I})
\end{aligned}
$$



따라서, $$\mathbf{w}$$에 대한 확률분포를 바탕으로 함수 $$\mathbf{f}$$에 대한 확률분포를 도출할 수 있습다.

$$
\mathbf{f}=\mathcal{N}(\mathbf{f}|\mathbf{0},\mathbf{K})
$$



여기서, 
$$
\mathbf{w}\sim\mathcal{N}(\mathbf{w}|\mathbf{0},\alpha^{-1})
$$
라 가정했고, $$\mathbf{f}$$는 $$\mathbf{w}$$에 대한 선형결합이므로 $$\mathbf{f}$$는 가우시안 분포입니다.


$$
\begin{aligned}
\mathbb{E}(\mathbf{f})&=\Phi\mathbb{E}(\mathbf{w})=\mathbf{0} \\
\mathrm{Cov}(\mathbf{f})&=\mathbb{E}(\mathbf{f}\mathbf{f}^T)=\Phi\mathbb{E}(\mathbf{w}\mathbf{w}^T)\Phi^T=\alpha^{-1}\Phi\Phi^T=\mathbf{K} \\
[\mathbf{K}]_{i,j}&=K_{ij}=k(\mathbf{x}_i,\mathbf{x}_j)=\alpha^{-1}\phi(\mathbf{x}_i)^T\phi(\mathbf{x}_j)
\end{aligned}
$$



## 2. Gaussian Process Regression

------

## **1. weight space view**

### 1. Bayesian formulation about linear regression

**정의**

Bayesian Linear Regression을 다시 생각해봅시다. 데이터가 $$\mathcal{D}=\{(\mathbf{x}_i,y_i)\}_{i=1}^n, X=[\mathbf{x}_1^T, \cdots, \mathbf{x}_n^T]$$로 주어지면, 

(1) function와 target은 다음과 같이 주어집니다.



$$
\mathbf{f}(\mathbf{x})=\mathbf{x}^T\mathbf{w}
$$


$$
\mathbf{y}(\mathbf{x})=\mathbf{f}(\mathbf{x})+\epsilon, \quad \epsilon \sim \mathcal{N}(0,\beta^{-1})
$$



(2) 그리고, likelihood, (weight의) prior는 다음과 같습니다.


$$
\begin{aligned} 
p(\mathbf{y}|X,\mathbf{w})&=\mathcal{N}(\mathbf{y}|X^T\mathbf{w},\beta^{-1}) \\
p(\mathbf{w})&=\mathcal{N}(\mathbf{0},\mathbf{\Sigma}_p)
\end{aligned}
$$



(3) 그렇다면, Bayesian rule을 이용해 posterior는 다음과 같이 구할 수 있습니다.


$$
\begin{aligned}
p(\mathbf{w}|\mathbf{y},X)&=\mathcal{N}(\beta A^{-1}X\mathbf{y},A^{-1}) \\
\text{where } A&=\beta XX^T+\mathbf{\Sigma}_p^{-1}
\end{aligned}
$$

(proof)



$$
p(\mathbf{w}|\mathbf{y},X) = p(\mathbf{y}|X,\mathbf{w})p(\mathbf{w})/p(\mathbf{y}|X)
$$



$$
\begin{align*}
&\log p(\mathbf{w}|\mathbf{y},X) \\
&\propto\ [-(\mathbf{y}-X^T\mathbf{w})^T\beta\mathbf{I}(\mathbf{y}-X^T\mathbf{w})][-\mathbf{w}^T\mathbf{\Sigma}_p^{-1}\mathbf{w}] \\
&\propto -\mathbf{w}^T(\beta XX^T+\Sigma_p^{-1})\mathbf{w} -\mathbf{y}^T\cdot\beta\mathbf{I}\cdot X^T\mathbf{w} - \mathbf{w}^TX\cdot\beta\mathbf{I}\cdot X^T\mathbf{w} \\
&=-(\mathbf{w}-\bar{\mathbf{w}})^TA(\mathbf{w}-\bar{\mathbf{w}}) \\
&\textrm{where } \bar{\mathbf{w}}=\beta A^{-1}X\mathbf{y}, A=\beta XX^T+\mathbf{\Sigma}_p^{-1}
\end{align*}
$$



(4) posterior를 구했으니, 이제 예측분포(predictive distribution)를 구할 수 있습니다.



$$
\begin{aligned}
p(\mathbf{y}_*|\mathbf{x}_*,X,\mathbf{y})&=\mathcal{N}(\beta X_*^TA^{-1}X\mathbf{y},\mathbf{x}_*^TA^{-1}\mathbf{x}_*) \\
\textrm{where } A&=\beta XX^T+\mathbf{\Sigma}_p^{-1}
\end{aligned}
$$



(proof)


$$
p(\mathbf{f}_*|\mathbf{x}_*,X,\mathbf{y}) =\int p(\mathbf{f}_*|X_*,\mathbf{w})p(\mathbf{w}|X,\mathbf{y})d\mathbf{w}
$$
를 풀기 위해, Gaussian distribution의 joint 분포를 conditional Gaussian distribution으로 계산할 수 있는 lemma를 사용합니다. 아래에서 다시 사용할 것이니 기억해주세요!

(lemma 1)


$$
\begin{aligned}
p(\mathbf{w})&=\mathcal{N}(\boldsymbol{\mu},\Lambda^{-1}) \\
p(y_*|\mathbf{w})&=\mathcal{N}(A\mathbf{w}+\mathbf{b},L^{-1}) \\
p(y_*)&=\mathcal{N}(A\boldsymbol{\mu}+\mathbf{b},L^{-1}+A\Lambda^{-1}A^T)
\end{aligned}
$$



를 이용하면,



$$
\begin{aligned}
p(\mathbf{w})&=\mathcal{N}(\bar{\mathbf{w}},A^{-1}) \\
p(\mathbf{y}_*|\mathbf{w})&=\mathcal{N}(\mathbf{x}_*^T\mathbf{w},\beta \mathbf{I}) \\
p(\mathbf{y}_*)&=\mathcal{N}(\mathbf{x}_*^T\bar{\mathbf{w}},\beta\mathbf{I}+\mathbf{x}_*^TA^{-1}\mathbf{x}_*)
\end{aligned}
$$



로 풀 수 있습니다.



### 2. kernel trick

자, 이제 3단원에서 했던 것들처럼, feature space로 mapping하는 basis function을 도입합니다. basis function은, $$\phi: \mathbb{R}^n \longrightarrow \mathbb{R}^N$$ : input space → high dim feature space (N>>n)로 정의하고,



(1) function를 다음과 같이 정의합니다. 1.의 (1)과 비슷한 형태입니다.



$$
\mathbf{f}(\mathbf{x})=\phi(\mathbf{x})^T\mathbf{w}
$$


$$
\phi(\mathbf{x})=\left(\phi(\mathbf{x}_1\right) \cdots \phi(\mathbf{x}_n)) \in \mathbb{R}^{N\times n}
$$



(2) likelihood와 weight의 prior는 1.의 (2)와 유사합니다.


$$
\begin{aligned} 
p(\mathbf{y}|X,\mathbf{w})&=\mathcal{N}(\mathbf{y}|\Phi^T\mathbf{w},\beta^{-1}) \\
p(\mathbf{w})&=\mathcal{N}(\mathbf{0},\mathbf{\Sigma}_p)
\end{aligned}
$$


(3) posterior도 1.(3)과 유사합니다.


$$
\begin{aligned}
p(\mathbf{w}|\mathbf{y},X)&=\mathcal{N}(\beta A^{-1}\Phi^T\mathbf{y},A^{-1}) \\
\text{where } A&=\beta \Phi\Phi^T+\mathbf{\Sigma}_p^{-1}
\end{aligned}
$$


(4) 이로부터 구할 수 있는 predictive distribution은 다음과 같습니다.



$$
\begin{aligned}\mathbf{f}_*|\mathbf{x}_*,X,\mathbf{y}&\sim\mathcal{N}(\beta \phi(\mathbf{x}_*)^TA^{-1}\Phi\mathbf{y},\phi(\mathbf{x}_*)^TA^{-1}\phi(\mathbf{x}_*)) \\
\textrm{where } A&=\beta\Phi\Phi^T+\mathbf{\Sigma}_p^{-1}, A \in \mathbb{R}^{N\times N}\end{aligned}
$$



(5) 여기서, N이 클 경우 $$A^{-1}$$를 계산하는 것은 굉장히 computationally incompatible합니다. 따라서, N>>n인 경우, matrix inversion formulation을 활용해 계산해야하는 역행렬 dim을 계산이 쉽게 바꿉니다.


$$
(Z+UW^{-1}V^T)^{-1}=Z^{-1}-Z^{-1}U(W^{-1}+V^TZ^{-1}U)^{-1}V^TZ^{-1}
$$

$$
A^{-1}=(\Sigma_p^{-1}+\beta\Phi\Phi^T)^{-1}=\Sigma_p+\Sigma_p\Phi(\beta\mathbf{I}+\Phi^T\Sigma_p\Phi)^{-1}\Phi^T\Sigma_p
$$



따라서, 계산해야하는 역행렬은 $$n\times n$$행렬인 $$(\beta\mathbf{I}+\Phi^T\Sigma_p\Phi)^{-1}$$ 입니다.



(6) 이제, kernel 을 정의할 수 있습니다. kernel function을 다음과 같이 정의합니다.


$$
\begin{aligned}
k(\mathbf{x},\mathbf{x}')&=\phi(\mathbf{x})^T\Sigma_p\phi(\mathbf{x}') \\
&=\phi(\mathbf{x})^T(UDU^T)^{1/2}(UDU^T)^{1/2}\phi(\mathbf{x}') \\
&=\psi(\mathbf{x})\psi(\mathbf{x}')
\end{aligned}
$$


그렇다면, Covariance function은 다음과 같이 정의할 수 있습니다.


$$
\mathbf{K}=\Phi^T\Sigma_p\Phi
$$



(7) 그렇다면, 여기서 kernel function을 이용해 구한 covariance matrix을 다음과 같이 나타낼 수 있습니다.


$$
\begin{aligned}\mathbf{f}_*|\mathbf{x}_*,X,\mathbf{y}&\sim\mathcal{N}\left(\beta\phi(\mathbf{x}_*)^TA^{-1}\Phi\mathbf{y}, \phi(\mathbf{x}_*)^TA^{-1}\phi(\mathbf{x}_*)\right) \\ &\sim\mathcal{N}\left(\phi_*\mathbf{\Sigma}_p \Phi(\mathbf{K}+\beta^{-1}\mathbf{I})^{-1}\mathbf{y}, \phi_*^T\mathbf{\Sigma}_p^{-1}\phi_*-\phi_*^T\mathbf{\Sigma}_p\Phi(\mathbf{K}+\beta^{-1}\mathbf{I})^{-1}\Phi^T\mathbf{\Sigma}_p\phi_*\right) \\ &\sim\mathcal{N}\left(k_*(K+\beta^{-1}\mathbf{I})^{-1}\mathbf{y}, k_{**}-k_*(K+\beta^{-1})^{-1}k_* \right) \end{aligned}
$$



(proof)

mean을 구하기 위해, 다음과 같은 process로 계산합니다.


$$
\begin{aligned}
A\mathbf{\Sigma}_p\Phi&=\beta\Phi(\mathbf{K}+\beta^{-1}\mathbf{I})=\mathbf{\Sigma}_p\Phi(\mathbf{K}+\beta^{-1}\mathbf{I})^{-1} \\
\beta A^{-1}\Phi&=\beta A^{-1}\Phi(\beta\cdot\mathbf{I}+\mathbf{K})(\beta\cdot\mathbf{I}+\mathbf{K})^{-1}\\
&=A^{-1}\cdot A\Sigma_p\Phi(\beta\cdot\mathbf{I}+\mathbf{K})^{-1}
\end{aligned}
$$


위를 이용해, predictive mean은,


$$
\begin{aligned}
\beta\phi(\mathbf{x}_*)^TA^{-1}\Phi\mathbf{y}&=\phi(\mathbf{x}_*)^T\Sigma_p\Phi(\beta\cdot\mathbf{I}+\mathbf{K})^{-1}\mathbf{y} \\
&=\mathbf{k}(X,\mathbf{x}_*)(\beta\cdot\mathbf{I}+\mathbf{k}(X,X))^{-1}\mathbf{y}
\end{aligned}
$$


다음으로, Covariance를 구합니다. 


$$
\begin{aligned}
&\phi(\mathbf{x}_*)^TA^{-1}\phi(\mathbf{x}_*)\\
&=\phi(\mathbf{x}_*)^T\Sigma_p\phi(\mathbf{x}_*)-\phi(\mathbf{x}_*)^T\Sigma_p\Phi(\beta\mathbf{I}+\Phi^T\Sigma_p\Phi)^{-1}\Phi^T\Sigma_p\phi(\mathbf{x}_*) \\
&=\mathbf{k}(\mathbf{x}_*,\mathbf{x}_*)-\mathbf{k}(X,\mathbf{x}_*)^T[\beta\mathbf{I}+\mathbf{k}(X,X)]^{-1}\mathbf{k}(X,\mathbf{x}_*)
\end{aligned}
$$


따라서, basis function을 이용한 bayesian linear regression에 kernel function을 도입하여 Gaussian Process Regression을 유도하였고 위의 식은 predictive distribution입니다. 이제, weight space에서 살펴본 GPR이 function space에서 본 GPR과 같음을 보일 것입니다.





## **2. function space view**

$$\mathbf{y}$$가 GP를 따르는 함수이면서 error(noise)가 포함되어 있다고 하면, 다음과 같이 정의할 수 있습니다.


$$
\begin{aligned}
\mathbf{y}(\mathbf{x})&=\mathcal{GP}(m(\mathbf{x}),k(\mathbf{x},\mathbf{x}')+\beta^{-1}) \\
\mathbf{y}&=\mathbf{f}+\boldsymbol{\epsilon}, \quad \epsilon_n\sim\mathcal{N}(0,\beta^{-1}),n=1,\cdots,N
\end{aligned}
$$


$$
\begin{aligned}
\mathbf{y}|\mathbf{f}&\sim\mathcal{N}(\mathbf{f},\beta^{-1}\mathbf{I}_N) \\
\mathbf{f}&\sim\mathcal{N}(\mathbf{0},\mathbf{K}), \quad \mathbf{K}=\alpha^{-1}\Phi\Phi^T
\end{aligned}
$$

### 1. Inference

GP를 위와 같이 정의하면, function $$\mathbf{y}$$는 다음과 같이 유도할 수 있습니다.


$$
\mathbf{y}\sim\mathcal{N}(\mathbf{y}|\mathbf{0},\mathbf{C}), \quad \mathbf{C}=\mathbf{K}+\beta^{-1}\mathbf{I}_N
$$



(proof)



여기서 우리는 위의 lemma 1 을 사용할 것입니다. 다시 언급하자면, 


$$
\begin{aligned}
p(\mathbf{x})&=\mathcal{N}(\mu,\Lambda^{-1})\\ p(\mathbf{y}|\mathbf{x})&=\mathcal{N}(A\mathbf{x}+b,L^{-1})
\end{aligned}
$$


이면,


$$
p(\mathbf{y})=\mathcal{N}(A\mu+b,L^{-1}+A\Lambda^{-1}A^T)
$$



입니다. 따라서, function $$\mathbf{y}$$는 다음과 같이 구할 수 있습니다.


$$
p(\mathbf{y})=\int p(\mathbf{y}|\mathbf{f})p(\mathbf{f})d\mathbf{y}=\mathcal{N}(\mathbf{0},\beta^{-1}\mathbf{I}_N+\mathbf{K})
$$



### 2. Prediction

predictive value(vector) $$\mathbf{f}_*$$ 의 분포는 새로운 input을 $$\mathbf{x}_*$$라 한다면, 다음과 같습니다.



$$
\begin{pmatrix} \mathbf{y} \\ \mathbf{y}_* \end{pmatrix} \sim \left( \begin{pmatrix} \mathbf{0} \\ 0 \end{pmatrix}, \begin{pmatrix} \mathbf{K}+\beta^{-1}\mathbf{I}_N & \mathbf{k}_* \\ \mathbf{k}_*^T & \mathbf{k}_{**}+\beta^{-1} \end{pmatrix} \right)
$$



where 
$$
\mathbf{k}_*=\mathbf{k}(\mathbf{x}_n,\mathbf{x}_*), \mathbf{k}_{**}=\mathbf{k}(\mathbf{x}_*,\mathbf{x}_*)
$$



$$
\mathbf{y}_*|\mathbf{y}\sim\mathcal{N}\left(\mathbf{k}^T(\mathbf{K}+\beta^{-1}\mathbf{I}_N)^{-1}\mathbf{y}, \mathbf{k}_{**}+\beta^{-1}-\mathbf{k}_*^T(\mathbf{K}+\beta^{-1}\mathbf{I}_N)^{-1}\mathbf{k}_*\right)
$$



(proof)



위를 계산하기 위해 다음과 같은 lemma 2 를 사용합니다. 이는 분할 행렬로 나타나진 joint Gaussian distribution에서 두 independent한 variables에 대한 conditional Gaussian distribution으로 나타내는 것입니다.


$$
p(\mathbf{x}_a|\mathbf{x}_b)=\mathcal{N}\left(\boldsymbol{\mu}_a+\mathbf{\Sigma}_{ab}\mathbf{\Sigma}_{bb}^{-1}(\mathbf{x}_b-\boldsymbol{\mu}_b), \mathbf{\Sigma}_{aa}-\mathbf{\Sigma}_{ab}\mathbf{\Sigma}_{bb}^{-1}\mathbf{\Sigma}_{ba}\right)
$$


이 lemma 2 를 이용하면, 다음과 같이 예측 평균과 분산을 구할 수 있습니다.



$$
\boldsymbol{\mu}_{\mathbf{y}_*|\mathbf{y}}=0+k_*^T(K+\beta^{-1}I)^{-1}(\mathbf{y}-0)=k_*C^{-1}\mathbf{y}
$$

$$
\Sigma_{\mathbf{y}_*|\mathbf{y}}=k_{**}+\beta^{-1}-k_*^T(K+\beta^{-1}I_N)^{-1}k_
*=k_{**}+\beta^{-1}-k_*^TC^{-1}k_*
$$



이렇게 구해진 predictive distribution은 위의 weight space view에서 본 predictive distribution과 같음을 확인할 수 있습니다.



여기서, 짚고 넘어가야 할 점이,


- $$\mathbf{C}$$는 positive definite이어야 합니다.

  - $$K$$의 eigen value ≥0 이면, $$k(x_i,x_j)$$가 모든 $$x_i,x_j$$에 대해 positive definite이게 됩니다.

- predictive mean은 다음과 같이 다시 적을 수 있습니다.

  $$
\mathbf{k}_*^T\mathbf{C}^{-1}\mathbf{y}=\sum_{n=1}^Na_n\mathbf{k}(\mathbf{x}_n,\mathbf{x}_*), \quad a_n=[\mathbf{C}^{-1}\mathbf{y}]_n
  $$
  
  
  - $$M<<N$$일 때 GP가 효율적임은 gram matrix 계산의 complexity로 인해 그렇습니다.



## **3. Hyper-parameter**

GPR에서 어떤 파라미터의 형태를 찾아보기는 어렵습니다. kernel function만 정의하면 되기 때문입니다. 하지만 kernel function 안에 들어가있는 hyper-parameter에 대한 값은 정해주어야 합니다. 이 hyper-parameter를 학습하기 위한 다양한 방법 중 여기서는 *marginal likelihood* 방법을 사용합니다. MLE에서 likelihood 를 최대화 하듯이, GPR에서는 Hyper parameter에 대한 'marginal' likelihood  $$p(\mathbf{y}|\theta)$$를 최대화 합니다.



$$
\log p(\mathbf{y}|\theta)=\frac{1}{2}\log|\mathbf{C}|-\frac{1}{2}\mathbf{y}^T\mathbf{C}^{-1}\mathbf{y}-\frac{N}{2}\log(2\pi)
$$

$$
\frac{\partial}{\partial\theta_i}\log p(\mathbf{y}|\theta)=-\frac{1}{2}Tr\left( \mathbf{C}^{-1} \frac{\partial C}{\partial\theta_i}\right)-\frac{1}{2}\mathbf{y}^T(-\mathbf{C}^{-1}\frac{\partial C}{\partial\theta_i}\mathbf{C}^{-1})\mathbf{y}
$$


- 위는 non-convex이므로 단순히 gradient = 0으로 계산할 수 없습니다. 따라서 conjugate gradient descent와 같은 방법으로 풉니다.
- fully Bayesian으로 $$\theta$$에 prior 주는 계산도 생각할 수 있지만 굉장히 어렵습니다.



## **4. ARD: Automatic Relevance Determination**

ARD는 각 input variable에 대해 다른 parameter를 사용하여 각 입력 값에 대해 상대적인 중요도(유용성)를 산출해내는 방법입니다. 



다음과 같은 커널 함수를 생각해봅시다.


$$
k(\mathbf{x},\mathbf{x}')=\theta_0\exp\{-\frac{1}{2}\sum_{i=1}^2\eta_i(x_i-x_i')^2\}
$$


- $$\eta_i$$ 값이 작아질수록 함수는 해당 input 변수 $$x_i$$에 대해 상대적으로 덜 민감하게 됩니다.
- 반대로 $$\eta_i$$ 값이 커지면 함수는 $$x_i$$에 민감하게 반응하여 함수값이 요동치게 됩니다.
- MLE를 적용하여 데이터로부터 이 파라미터를 조절하면 예측 분포에 적게 영향을 미치는($$\eta_i$$값이 작은) input 변수를 찾을 수 있습니다. 이렇게 영향을 적게 미치는 입력 변수를 제거할 수 있습니다. 즉, ARD를 이용해 변수를 선택할 수 있습니다!





## **5. Code Implementation**

여기까지, GPR이었습니다. 이제 이를 Python 코드로 작성해보겠습니다. 주로 numpy를 이용해 작성하였고 능력 부족으로 complexity는 고려가 거의 안되었습니다.(...ㅠ) 혹시 코드 구현에 대해 좋은 의견이 있으신 분은 조언 부탁드립니다!



(1) 먼저, 세팅을 한 후,

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

## Settings
# load data
X, y = load_data_function()
num_outputs = y.shape[1]
num_inputs, num_features = X.shape
scaler = StandardScaler() # or MinMaxScaler()
X = scaler.fit_transform(X)
# initial hyper-parameter(s): scala or list
theta = 1
noise = 1 # noise variance(inverse beta)
```



(2) 모델을 피팅합니다. stable한 computation을 위해 NLL 계산 중 cholesky decomposition을 사용합니다.

```python
## Fitting
# distances for kernel function
def distance(x1,x2):
	return np.dot(np.subtract(x1,x2), np.subtract(x1,x2)) # Gaussian kernel

def distance_matrix(mat1, mat2, same=False):
	n1,n2 = mat1.shape[0],mat2.shape[0]
	dist = np.zeros([n1,n2], dtype=np.float64)
    if same:
        for i in range(n1):
            dist[i,i:] = np.array(list(map(lambda j: distance(mat1[i,:],mat2[j,:]), range(i, n2))))
        dist = np.maximum(dist, dist.T)
    else:
        for i in range(n1):
            dist[i,i:] = np.array(list(map(lambda j: distance(mat1[i,:],mat2[j,:]), range(n2))))
    return dist

dist = distance_matrix(X, X, same=True)

# Gram Matrix
def gram_matrix(dist, theta):
	return np.exp(- dist / theta)
log_func = np.vectorize(lambda x: np.log(x))

K = gram_matrix(dist, theta)
K_f = K + noise * np.identity(K.shape[0])

# Negative Log Likelihood
L = np.cholesky(K_f)
a = linalg.solve_triangular(L.T, linalg.solve_triangular(L, y, lower=True))
nll = np.dot(y.T,a) /2 + log_func(L.diagonal()).sum() + num_inputs * np.log(2*np.pi) /2
```



(3) 여기서, NLL을 이용해 hyper-parameter를 learning합니다. NLL을 minimize하기 위한 optimization 방법은 주로 gradient descent 혹은 conjugate gradient descent를 사용합니다. 여기 코드에는 포함시키지 않았습니다.

```python
## Learning Hyper-parameters using NLL
```



(4) 예측합시다. 여기서 stable computation을 위해 gauss elimination을 사용합니다.

```python
## Prediction
# load data
X_pred, y_pred = load_data_function()
X_pred = scaler.transform(X_pred)

# predictive kernel functions
dist_pred = distance_matrix(X, X_pred)
dist_pred_pred = distance_matrix(X_pred, X_pred, same=True)

k_pred = gram_matrix(dist_pred , theta)
k_pred_pred = gram_matrix(dist_pred_pred , theta)

# predictive mean
pred_mean = np.dot(k_pred.T, a)

# predictive covariance
v = linalg.solve_triangular(L, k_pred)
pred_cov = k_pred_pred - np.dot(v.T, v) + noise
```







# 3. Gaussian Process Classifier

------









###### Reference

1. Bishop, C. M. (2006). *Pattern recognition and machine learning*. springer.
2. Williams, C. K., & Rasmussen, C. E. (2006). *Gaussian processes for machine learning* (Vol. 2, No. 3, p. 4). Cambridge, MA: MIT press.
3. 최성준 교수님, edwith, *Bayesian Deep Learning*