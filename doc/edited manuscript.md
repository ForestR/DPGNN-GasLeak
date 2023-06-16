## 2.  Problem statement

Our task is to improve the accuracy and reliability of automated leakage detection and localization and demonstrate it by comparing it with monitored time-series data collected by spatially-distributed sensors.  Existing GNN-based approaches are based on the 'point-estimation' approach and cannot handle uncertainty of detection results.  In addition, such approaches locate the leakage position using the single dependency weights among sensors, which are calculated using graph attention learning.  The pressure data fluctuates steadily under normal conditions in the natural gas transmission and distribution pipeline system.  Once a leak occurs, the loss of fluid will change the fluid density near the leakage location and a rapid pressure drop can be seen.  The pressure difference forces the natural gas to squeeze towards the leakage location, establishing a new pressure gradient in the leakage region.  The pressure variation in the leakage location is larger compared to areas distant from the leak.  Due to the unavoidable stochastic background noise in the pipe network, the pressure fluctuations decay with increasing distance (Gupta et al., 2018).  However, the aforementioned dependency weights calculated by prevalent "point-estimation'-based GNN methods cannot represent pressure variation recorded by sensors near the leakage location.   
   
We aim to model the posterior distribution of dependency weights to present the larger pressure variation nearby and accordingly locate the leakage more accurately by integrating variational Bayesian inference with attention-based GNN.  By variational Bayesian inference, we also provide the additional uncertainty of leakage detection.  From a mathematical perspective, this is to say we need to model the probability density $P(A|X)$ at time $t$. $A$ is dependency weight among spatially-distributed sensors and $X$ is the previous $s$-steps pressure data monitored by sensors, which can be expressed as:

$$
A = \begin{bmatrix}
\alpha_{1,1} & \cdots & \alpha_{1,J} \\
\vdots & \ddots & \vdots \\
\alpha_{m,1} & \cdots & \alpha_{m,J}
\end{bmatrix}
\quad (1)
$$

$$
X = \begin{bmatrix}
x_{1,t-s} & \cdots & x_{1,t-1} \\
\vdots & \ddots & \vdots \\
x_{m,t-s} & \cdots & x_{m,t-1}
\end{bmatrix}
\quad (2)
$$

where $m$ is the number of sensors, $J$ is the number of neighbor sensors of the target sensor.

## 3. Proposed approach

Fig. 1 demonstrates the architecture of the proposed variational Bayesian inference graph attention neural network, namely VB_GAnomaly, for real-time automated leakage detection and localization in complex pipe networks.

Fig. 1 Architecture of VB_GAnomaly. (
- Inputs: Time-series datasets in normal state
    1. Association Graph Construction
    2. Variational Bayesian-based Graph Attention learning
    3. Multilayer perceptron
- Outputs: leakage detection and localization with uncertainty
)


### 3.1 Attention-based graph neural network as the backbone
The attention-based graph neural network is applied as a backbone because the attention mechanism could capture the dependency among spatially-distributed sensors accurately (Choi et al., 2021).

(1) The first part of our proposed VB_GAnomaly is an association graph presenting the connections among sensors of the pipeline network (Hao et al., 2022; Li et al., 2023). Such connections depend on the flow state such as flow direction, speed etc., inside the pipeline network. We then construct the association graph structure $G= (V, E)$ with nodes $V$ presenting sensors and edges $E$ presenting the connections among sensors. We denote the number of sensors as $N_{nodes}= m$ and the number of connections as $N_{edges}= n$. We further define the adjacency matrix $𝐴𝑑_{𝑖𝑗}$ presenting whether connection between sensor $i$ and $j$ exists. In addition, a linear embedding operation is applied to calculate the feature vector $𝑣𝑖$ of time-series data from sensor $i$, which can be expressed as:

$$
v_i = \mathrm{Embedding}^{w_e}([x_{i,t-s},x_{i,t-s+1},\cdots,x_{i,t-1}]) \quad (3)
$$

where $i \in \{1,2,\cdots,m \}$, $w_e$ denotes the parameters in the linear embedding neural layer.

By integrating $v_i$ with time-series data $X_i$, the initial node representation corresponding to each sensor $\gamma_i$ can be calculated as:

$$
\gamma_i = v_i \oplus w_g X_i \quad (4)
$$

where $\oplus$ denotes matrix concatenation and $w_g$ is the weight matrix.

(2) The second part of our proposed VB_GAnomaly is to quantify the connections among sensors by using attention-based neural network. We first apply $\mathrm{LeakyReLU}$ as the nonlinear activation to determine dependency weights $\xi(i,j)$ between sensor $i$ and its neighbor sensor:

$$
\xi(i,j) = \mathrm{attention}(i,j) = \mathrm{LeakyReLU}[w_a\top(\gamma_i \oplus \gamma_j)] \quad (5)
$$

where $𝑤_𝑎$ is weight vector of learned dependency weights for the attention mechanism, and $\gamma_𝑖$, $\gamma_𝑗$ are the initial node representation of sensor $i$ and sensor $j$.

Then, we apply the softmax function to normalize the dependency weights.

$$
a_{i,j} = \mathrm{softmax}(\xi(i,j)) = \frac{\exp(\xi(i,j))}{\sum_{k \in \mathcal{N}(i) \cup \{i\}}\exp(\xi(i,k))} \quad (6)
$$

where $a_{i,j}$ is normalized dependency weights.

The normalized dependency weights of sensor itself and the weights between sensor and its neighbor sensors are then integrated with feature vector of time-series data $v_i^t$ from sensor $i$, which can be expressed as:

$$
h_i = \mathrm{ReLU} (\alpha_{i,i}w_gv_i + \sum_{j \in \mathcal{N}(i)}\alpha_{i,j}w_gv_j) \quad (7)
$$

where $ℎ_𝑖$ is learned node representation.

Subsequently, the set of learned node representations $ℎ$ for all nodes can be expressed as:

$$
h = \{h_1,h_2,\cdots,h_m\} \quad (8)
$$

Then, $ℎ$ is regarded as input to a multilayer perceptron layer (MLP) to forecast the pressure $𝑌 = \{𝑌_1,𝑌_2,…,𝑌_𝑚\}$.

$$
Y = \mathrm{MLP}^{w_f}(h) \quad (9)
$$

where $𝑤_𝑓$ denotes the parameters in the MLP neural layer.

### 3.2 Variational Bayesian Inference

Variational Bayesian inference is then applied to model the posterior distribution density of dependency weights $P(A|X)$, where $A$ is dependency weight among spatially-distributed sensors and $X$ is the previous $s$-steps data monitored by sensors. Given $\chi = (X,A)$, the probability density $P(A|X)$ can be expressed as:

$$
P(A|X) = \int P(A|X,w)P(w)dw, \quad w \sim P(w|\chi) \quad (10)
$$

where $w$ presents a set of parameters in the Linear embedding neural layer and the graph attention deep neural network, $w = \{w_e,w_a,w_g\}$. $P(A|X,w)$ denotes the conditional probability density of the dependency weight $A$ given pressure sequence $X$ as well as the parameters $w$ of the deep learning neural network.

According to Bayesian theory, the log probability of $P(w|\chi)$ can be inferenced as:

$$
\log P(w|\chi) = \log \left(\frac{P(\chi|w)P(w)}{P(\chi)}\right) \quad (11)
$$

$P(\chi|w)$ is the likelihood of $\chi$ given $w$, $P(w)$ is the prior probability of initially assumed a list of $w$ values, $P(\chi)$ is the marginal probability.

By using the variational Bayesian inference, an approximate density distribution $Q_\epsilon(w)$ can be found to represent the posteriori probability $P(w|\chi)$. In order to measure the approximation between $Q_\epsilon(w)$ and $P(w|\chi)$, KL divergence is introduced to describe the similarity between both probability distributions, as illustrated below:

$$
\begin{aligned}
KL(Q_\epsilon(w)||P(w|\chi)) &= \int Q_\epsilon(w) \log \left( \frac{Q_\epsilon(w)}{P(w|\chi)} \right) dw \\
&= \int Q_\epsilon(w) \left( \log Q_\epsilon(w) - \log \left( \frac{P(\chi|w)P(w)}{P(\chi)} \right) \right) dw \\
&= \int Q_\epsilon(w) \left( \log Q_\epsilon(w) - \log P(\chi|w) + \log P(w) - \log P(\chi) \right) dw \\
&= \log P(\chi) - \int Q_\epsilon(w) \log \left( \frac{P(\chi|w)P(w)}{Q_\epsilon(w)} \right) dw  \quad (12)
\end{aligned}
$$

Since we aim to minimize the $\mathrm{KL}(Q_\epsilon(w)||P(w|\chi))$, and $\log P(\chi)$ is a constant that depend on the determined dataset, thus, minimizing the $\mathrm{KL}(Q_\epsilon(w)||P(w|\chi))$ is equivalent to maximizing the second terms of right side. Moreover, given that $\mathrm{KL}$ divergence is greater than or equal to 0 (0 if and only if $Q_\epsilon(w)$ is equal to $P(w|\chi)$), it can be deduced that the second terms of the right side is a lower bound of $\log P(\chi)$, donated as, $\mathrm{ELBO}$ (Evidence Lower Bound).

$$
\begin{aligned}
\mathrm{ELBO} &= \int Q_\epsilon(w) \log \left( \frac{P(\chi|w)P(w)}{Q_\epsilon(w)} \right) dw \\
&= \int Q_\epsilon(w) \log P(\chi|w) dw - \int Q_\epsilon(w) \log \left( \frac{Q_\epsilon(w)}{P(w)} \right) dw \\
&= \mathbb{E}_{w \sim Q_\epsilon(w)}[\log P(\chi|w)] - \mathrm{KL}(Q_\epsilon(w)||P(w))  \quad (13)
\end{aligned}
$$

where the first term of the right hand is the expectation term of $\log P(\chi|w)$ and the second term of the right hand is the KL divergence between $Q_\epsilon(w)$ and $P(w)$. Therefore, the variational distribution $Q_\theta(w)$ can be estimated by maximizing the expectation term of $\log P(\chi|w)$ and minimizing the KL divergence term $\mathrm{KL}(Q_\epsilon(w) \parallel P(w))$. Maximizing the expectation term of $\log P(D|w)$ and minimizing MSE between the predicted and observed pressure values, the first term of the right hand can be expressed as:

$$
\mathbb{E}_{w \sim Q_\epsilon(w)}[\log P(\chi|w)] = -\mathrm{MSE}(X,Y) \quad (14)
$$

Assuming any distribution could be modeled by multi-Gaussian mixture distributions, we determine $Q_\epsilon(w) = P\mathcal{N}(\theta,\sigma^2I) + (1 - P)\mathcal{N}(0,\sigma^2I)$ and $P(w) = \mathcal{N}(0,\sigma^2I)$. Then, the KL divergence term $\mathrm{KL}(Q_\epsilon(w) \parallel P(w))$ can be expressed as:

$$
\mathrm{KL}(Q_\epsilon(w) \parallel P(w)) \approx \frac{\rho}{2} \epsilon^T\epsilon - C_1\sigma^2 - C_2\ln \sigma^2 + C_3 \quad (15)
$$

where $\rho$ is the pre-defined dropout probability, $\epsilon$ is the optimized variational parameter, $C_1$, $C_2$ and $C_3$ are constants.

Eventually, the loss function can be minimized by maximizing the expectation term of $\log P(\chi|w)$ and minimizing the KL divergence term $\mathrm{KL}(Q_\epsilon(w) \parallel P(w))$:

$$
\mathrm{LOSS} \approx -\mathrm{MSE}(X,\mathrm{VB\_GAnomaly}_{w \sim Q_\epsilon(w)}(X)) + \frac{\rho}{2} \epsilon^T\epsilon - C_1\sigma^2 - C_2\ln \sigma^2 + C_3  \quad (16)
$$

A stochastic gradient descends (SGD) optimization algorithm is applied to minimize the loss function for the determination of variational distribution $Q_\epsilon(w)$.

### 3.3 Leakage Detection and Localization

Given the new testing input $X^*$, the posterior distribution $P(A^*|X^*)$ can be approximated by $w \sim Q_\epsilon(w)$, as expressed in Eq. (17):

$$
P(A^*|X^*) = \int P(A^*|X^*,w)Q_\epsilon(w)dw, \quad w \sim Q_\epsilon(w) \quad (17)
$$

Further, the Kernel density estimation (KDE) are used to approximate the probability density function (PDF) of the dependency weights $A^*$ (He & Zhang, 2020). Numerous dependency weights can be computed by Monte Carlo (MC) sampling parameters $w$ from the variational distribution $Q_\epsilon(w)$ in the deep learning of the Eq. (3)- Eq. (6). Accordingly, the predicted pressure can be expressed as:

$$
Y^{*1},Y^{*2},\dots,Y^{*mc} = \mathrm{VB\_GAnomaly}^{w^1,w^2,\dots,w^{mc}}(X^*), \quad w^1,w^2,\dots,w^{mc} \sim Q_\epsilon(w) \quad (18)
$$

where, $mc$ represents the pre-defined sampling number of our proposed model. $Y^{*mc}$ is the predicted pressure of all sensors under $mc$-th sampling parameter $w^{mc}$.

For each predicted pressure $Y_i^{*z}$ of sensor under $z$-th sampling parameter $w^z$, a proportional error $Er_{i}^{*z}$ can be computed comparing to the observed pressure $X_i'$, as expressed below:

$$
Er_{i}^{*z} = \frac{|X_i' - Y_{i}^{*z}|}{X_i'}  \quad (19)
$$

The mean value of the proportional deviation under $mc$ sampling is calculated as the deviation $Z_i$ of sensor $i$, as expressed below:

$$
Z_i = \frac{1}{mc} \sum_{z=1}^{mc} Er_{i}^{*z} \quad (20)
$$

By comparing the deviation of each sensor, the sensor $\eta^t$ with the maximum deviation at time $t$ can be obtained. The $mc$ proportional errors of the sensor $\eta^t$ are sorted according to their values, and the median value of sensor $\eta_t$’s proportional error $Er_{\eta^t}^*$ is donated as ${Me\_Er}^*$:

$$
{Me\_Er}^* = \frac{Er_{mc/2} + Er_{mc/2 + 1}}{2} \quad (21)
$$

where $Er_{mc}$ is the ${mc/2}$-th proportional error after sorting.

Then, the proportional error of sensor $\eta^t$ is compared with the normal threshold to determine whether time $t$ is in an anomaly, and the normal threshold $Th(i)$ of sensor $i$ generated by the maximum deviation in validation dataset, which is normal data without label.

Thus, the leakage detection results can be obtained by comparing ${Me\_Er}^*$ with the normal threshold $Th(\eta^t)$ of sensor $\eta^t$, which is generated by the maximum deviation in the validation dataset without label. The leakage detection result is expressed as a set of binary labels indicating whether time $t$ is leaking or not, i.e. $B \in \{0,1\}$, where $B = 1$ indicates that time $t$ is leaking. And the detection result $B$ is expressed as:

$$
B = \begin{cases}
0 & \text{if } -Th(\eta^t) \leq {Me\_Er}^* \leq Th(\eta^t) \\
1 & \text{if } {Me\_Er}^* > Th(\eta^t) \text{ or } {Me\_Er}^* < -Th(\eta^t)
\end{cases} \quad (22)
$$

where $-Th(\eta^t)$ is the lower of the normal interval and $Th(\eta^t)$ is the upper of the normal interval.

Given $mc$ sampling proportional errors of sensor $\eta^t$, $mc$ leakage detection results can be obtained. Our model also gives a probabilistic result of distribution $P(B)$ at time $t$, which can be expressed as:

$$
P(B) = \left(\frac{\beta}{mc}\right)^B \left(1 - \frac{\beta}{mc}\right)^{1-B} \quad (23)
$$

where $\beta$ is the number of anomaly, $0 \leq \beta \leq mc$.

Once an anomaly is detected, the leakage can be positioned according to the uncertainty dependency weights. The number of maximum deviations for sensor $i$ during the leakage time $T$ is donated as $N_{di}$:

$$
N_{di} = \sum_{t=1}^T (\eta^t = i) \quad (24)
$$

Then, given sensor $\phi$ as the sensor with the maximum $N_{di}$, the standard deviation of dependency weights between the sensor $\phi$ and the neighbor sensor $j$ is output as $Std_{\phi,j}$,

$$
Std_{\phi,j} = \sqrt{\frac{\sum_{r=1}^{mc}\sum_{t=1}^T (\alpha^{*t,r}_{\phi,j} - \alpha^{*mean}_{\phi,j})^2}{mc \times T}} \quad (25)
$$

Therefore, a maximum standard deviation $Std_{\phi,\zeta}$ among all the neighboring sensors is calculated, as expressed:

$$
Std_{\phi,\zeta} = \max \{Std_{\phi,1}, Std_{\phi,2}, \dots, Std_{\phi,j}\} \quad (26)
$$

Thus, $Std_{\phi,\zeta}$ indicating that the position of the edge with the largest weight fluctuation is between nodes $\phi$ and $\zeta$, that is, the leak is located between sensor $\phi$ and sensor $\zeta$.

### 3.4 Evaluation metrics for leakage detection and localization

In this work, we use precision (Prec), recall (Rec), F1-score (F1), the area under the receiver operating characteristic curve (AUC) and the overall positioning accuracy (PAc) as the evaluation metrics (Ding et al., 2023). Given the confusion matrix of the predicted results and the true labels of leakage detection, we denote true positive samples, true negative samples, false positive samples, and false negative samples as TP, TN, FP, and FN, respectively. Precision indicates the percent of positive anomalies to all detected anomalies. Recall is the percent of correctly detected anomalies to all actual anomalies. F1-score can show the trade-off between the value of precision and recall regarding the positive anomalies. Precision, recall and F1-score can be calculated following Eq. (27)–Eq. (29).

$$
\mathrm{Prec} = \frac{TP}{TP + FP} \quad (27)
$$

$$
\mathrm{Rec} = \frac{TP}{TP + FN} \quad (28)
$$

$$
F1 = 2 \frac{\mathrm{Prec} \cdot \mathrm{Rec}}{\mathrm{Prec} + \mathrm{Rec}} \quad (29)
$$

Given $K$ as the number of positive samples and $N$ as the number of negative samples, AUC can be calculated as:

$$
\mathrm{AUC} = \frac{\sum_{ins_i \in positiveclass} rank_{ins_i} - K(K+1)/2}{KN} \quad (30)
$$

where $positiveclass$ is the set of order numbers of positive samples, and $rank_{ins_i}$ is order number of the $i$-th sample.

In addition, the overall positioning accuracy is measured by the percentage of leakage that has been correctly localized, as expressed below:

$$
\mathrm{PAc} = \frac{1}{n} \sum_{i=1}^n L(E_i,E_i) \quad (31)
$$

where $L(E_i,E_i)$ is the number of samples with a leakage position $E_i$ that are localized in pipeline $E_i$, $n$ is the number of all the pipelines.

## 4. Benchmark dataset

### 4.1 Experimental configuration

A natural gas leakage experiment system of urban gas transmission and distribution pipeline network is used to simulate gas flow with/without leakage. 

Fig. 2 Lab-scale experimental system of urban gas transmission and distribution pipeline leakage simulation.(
Fig.2 demonstrates the experimental system, which mainly consists of main pipeline and pipeline branches. The diameter of the main pipelines is 80mm, while the branch pipelines are of diameter 50mm. A gas regulator and several valves are installed in the pipeline network. The gas regulator maintains the inside pressure at 0.1 MPa. 5 ball valves installed at 5 pipeline positions are used to generate 5 gas leakages, namely leak1, leak2, leak3, leak4 and leak5. In addition, pressure signals which are convenient and stable indicators in pipeline leakage diagnosis (Zheng et al., 2020), are used as benchmark time-series data. Four pressure sensors, denoted as P1, P2, P3, and P4 are installed along the main pipeline and branches to collect the benchmark time-series data. 
)

The location of each leakage position and the installed sensor are summarized in Table 1.

Table 1: Location of the leakage point.

| Leak  | Upstream sensor | Downstream sensor | Location   |
|-------|-----------------|-------------------|------------|
| Leak1 | P3              | P4                | P3-P4      |
| Leak2 | P1              | P3                | P1-P3      |
| Leak3 | P1              | P2, P3            | P1-P2/P3   |
| Leak4 | P1              | P2                | P1-P2      |
| Leak5 | P1              | P2, P3            | P1-P2/P3   |

Finally, an online data processing system (DPS) is used to collect the online monitoring time-series pressure signals. The topology graph is shown in Fig. 3.

Fig. 3 Topology graph of experimental pipeline network.(
The topology graph of experimental pipeline network is constructed with nodes presenting sensors and edges presenting the pipeline connection between sensors, as introduced in section 3.1. 
)

By using such an experimental system, we monitor the inside pressure for 1 hour and collect $3600=3600s \times 1$ pressure values from each sensor with pressure sampling interval of 1s. Then, we divide all the pressure values into 3590 sequences, each of which includes 10 pressure values. Finally, we generate the benchmark training dataset $X_{train} \in \mathbb{R}^{(4,3590,10)}$, without leakage from 4 sensors. For benchmark testing dataset construction, we simulate 5 leakage scenarios by considering 5 leakage positions. For each scenario, we first monitor the inside pressure without leakage for 80s and then set the occurrence of leakage lasting for 80s. Then, we collect $800=5 \times 160$ pressure valves, which are divided with 790 sequences and each sequence includes 10 pressure values. Finally, we generate the benchmark testing dataset $X_{test} \in \mathbb{R}^{(4,790,10)}$ from 4 sensors.

An example of time-series pressure data monitored by 4 sensors under leak2 is presented in Fig. 4. 

Fig. 4 Time-series pressure data from 4 sensors under pressure 0.1MPa and leak2.(
As can be seen, before the occurrence of the leakage, the pressure fluctuates steadily. However, a rapid drop is observed once the leakage is initiated. Among all the sensors, the pressure of P3 shows a larger variance among all sensors, and reaches the lowest pressure of 0 due to its closer location to the leakage position.
)

### 4.2 Benchmark dataset processing

Since the experimental pipeline network is composed of pipelines with different pressure grades, i.e., DN50 and DN80, the variations of the monitoring time-series pressure signal among different sensors has a great discrepancy. In this regard, data processing is required to ensure all the monitoring data with the same magnitude in order to accelerate the model’s convergence and generalization capability. The minmax normalization approach is adopted to normalize all the time-series data $X$ between 0 and 1 as expressed (Zheng et al., 2020):

$$
x_n = \frac{x_l - x_{min}}{x_{max} - x_{min}} \quad (32)
$$

where $x_n$ is the normalized value, $x_l$ is the original value, $x_{max}$ and $x_{min}$ are the maximum and minimum of the original values.


An example of time-series pressure data monitored by 4 sensors under leak2 is presented in Fig. 4. 

Fig. 4 Time-series pressure data from 4 sensors under pressure 0.1MPa and leak2.(
As can be seen, before the occurrence of the leakage, the pressure fluctuates steadily. However, a rapid drop is observed once the leakage is initiated. Among all the sensors, the pressure of P3 shows a larger variance among all sensors, and reaches the lowest pressure of 0 due to its closer location to the leakage position.
)

## 5. Probabilistic graph deep learning model development

Our VB\_GAnomaly model is compiled by using Python version 3.6 and PyTorch version 1.5.1 with computer server of Intel(R) Xeon(R) Silver 4214R CPU @ 2.40GHz 2.39 GHz and 4 NVIDIA GeForce RTX 2080Ti GPU. Increasing Monte Carlo samples induces the posterior distribution of predicted pressure by Eq.(18) converged, which however increases the computational burden and harms our model’s real-time capability for decision-making. Therefore, the trade-off between our model’s accuracy and real-time capability should be determined by determining the optimal Monte Carlo $mc$ sampling number.

Taking experimental scenario of Leak 1 under inside pressure of 0.1 MPa as example, we first extract 10 groups of weights from variational distribution $Q_\epsilon(w)$ and accordingly calculate 10 groups of pressure values by Eq.(18). For each group, we extract the weights $mc$ times, and then determine $mc$ pressure values. This indicates each group includes $mc$ predicted pressure values. Fig. 5 displays 10 groups of Cumulative Density Function (CDF) curves of predicted pressure values under $mc=50$, $300$ and $500$, respectively. 

Fig. 5 CDF curves of 10 groups of predicted pressure value corresponding to sensor1 under $mc=50$, $300$ and $500$.(
From it, one can see an obvious decrease of the uncertainty interval of predicted pressure value from 0.010564 to 0.005367, as the number of MC samples increases from 50 to 300 when CDF value is 0.5. With further increasing the $mc$ number from 300 to 500, such decrease becomes negligible. However, increasing the MC samples would harm our model’s efficiency. In this regard, $mc=300$ is selected as the optimal for the trade-off between model’s accuracy and efficiency.
)

Fig. 6 PDF distribution of predicted pressure under $mc=300$ corresponding to sensor1.(
Fig. 6 displays the probability density function (PDF) distribution of the predicted pressure values of Leak 1 under inside pressure of 0.1 MPa. From it, one can see distribution of the predicted pressure follows the Gaussian distribution with a mean pressure value $mu=0.0977$ MPa and standard deviation $std=0.0288$. The difference between the mean pressure value $mu=0.0977$ MPa and the experimental pressure value 0.10 MPa is 0.0023, indicating our model’s accuracy under $mc=300$.
)

---
A general outline of the steps that could be taken to develop the proposed method using PyTorch:

1. Install and set up PyTorch and any required dependencies on your development environment.
2. Preprocess the benchmark dataset as described in section 4.2 of the paper to normalize the time-series data.
3. Implement the attention-based graph neural network as described in section 3.1 of the paper using PyTorch’s built-in neural network modules and functions.
4. Implement the variational Bayesian inference component as described in section 3.2 of the paper using PyTorch’s probabilistic programming capabilities.
5. Integrate the attention-based graph neural network and variational Bayesian inference components to create the VB_GAnomaly model as described in section 3.3 of the paper.
6. Train the VB_GAnomaly model on the preprocessed benchmark dataset using PyTorch’s optimization and training capabilities.
7. Evaluate the performance of the trained VB_GAnomaly model on a test dataset using the evaluation metrics described in section 3.4 of the paper.

This is just a high-level outline of the steps that could be taken to develop the proposed method using PyTorch. Each step would involve more detailed implementation decisions and choices based on your specific needs and requirements.

---
Create a repository in github to repeat the work in the paper "Towards deep probabilistic graph neural network for natural gas leak detection and localization without labeled anomaly data". 
Here is a possible tree structure for organizing the repository:
```
DPGNN-GasLeak/
├── data/
│   ├── raw/
│   └── preprocessed/
├── src/
│   ├── preprocessing/
│   ├── model/
│   └── training/
├── results/
├── LICENSE
└── README.md
```

Here is a README.md file for the repository.
```
# DPGNN-GasLeak (Deep Probabilistic Graph Neural Network for Natural Gas Leak Detection)

This repository contains code and data for replicating the work in the paper '[Towards deep probabilistic graph neural network for natural gas leak detection and localization without labeled anomaly data](https://doi.org/10.1016/j.eswa.2023.120542)' by Zhang et al. (2023). The paper proposes a deep probabilistic graph neural network that integrates an attention-based graph neural network and variational Bayesian inference to model spatial sensor dependency and localize natural gas leaks.

## Getting Started

These instructions will help you set up the project on your local machine for development and testing purposes.

### Prerequisites

- Install [PyTorch](https://pytorch.org/) and any required dependencies on your development environment.

### Data Preprocessing

Preprocess the benchmark dataset as described in section 4.2 of the paper to normalize the time-series data.

### Model Implementation

1. Implement the attention-based graph neural network as described in section 3.1 of the paper using PyTorch’s built-in neural network modules and functions.
2. Implement the variational Bayesian inference component as described in section 3.2 of the paper using PyTorch’s probabilistic programming capabilities.
3. Integrate the attention-based graph neural network and variational Bayesian inference components to create the VB_GAnomaly model as described in section 3.3 of the paper.

### Training

Train the VB_GAnomaly model on the preprocessed benchmark dataset using PyTorch’s optimization and training capabilities.

### Evaluation

Evaluate the performance of the trained VB_GAnomaly model on a test dataset using the evaluation metrics described in section 3.4 of the paper.

## License

This project is licensed under the [GPL License](LICENSE).
```

---




