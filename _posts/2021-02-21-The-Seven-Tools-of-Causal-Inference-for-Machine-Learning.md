---
title: The Seven Tools of Causal Inference for Machine Learning
author: euphoria0-0
date: 2021-02-21 18:00:00 +0800
categories: [Causal Learning, Paper Review]
tags: [Machine Learning, Paper Review, Causal Learning]
toc: true
math: true
comments: true
---



이 글에서는 Causal Inference에 대한 소개를 Judea Pearl 교수님의 논문(리포트)인 "The Seven Tools of Causal Inference with Reflections on Machine Learning"을 인용하여 설명하고자 합니다.



## 1. Intro

현재 많은 머신러닝 방법들의 가시적인 성과들이 나타나고 있습니다. 그러나 많은 응용 단계에서 근본적인 장벽을 가지고 있습니다. 크게, adaptability 혹은 robustness, explainability, understanding of cause-effect connections로 이야기할 수 있습니다. adaptability는 머신러닝이 학습되지 않은 새로운 상황에 대해 인식하고 대응하는 능력이 많이 부족하다는 것을 의미하고, explainability는 black box라는 머신러닝의 큰 문제점을 의미하는데, 모델이 예측한 사항의 이유를 설명할 수 없어 신뢰도가 낮은 것을 의미합니다. 그리고 이 논문에서 집중하는 것은 "Cause-Effect"입니다. Judea Pearl이 몇십년간 집중해온 연구 분야이기도 하고, 기존의 statistical correlation이 아닌 causal relationships의 필요성을 강조하는 분야입니다. ~~블로그 필자인 저도 많은 관심을 가지고 연구를 위해 공부하고 있는 분야입니다.~~

cause-effect connection을 이해하는 것은 매우 중요합니다. 인간 수준의 지능을 달성하는 데에 꼭 필요한 요소이며, 기존의 (statistical) machine learning으로는 다룰 수 없었던 cause에 대한 effect를 추론합니다. 특히, 기존에 관측한 데이터로는 알 수 없는 질문인 "What if?"에 대한 답을 다룰 수 있습니다. 이는, 미래적 질문인 "What if I make it happen?"와 회고적 질문인 "What if I had acted differently?"에 대한 것입니다. Pearl은 이 causal model로 이러한 질문에 답할 수 있다고 주장하는 연구를 해왔습니다.

이 글에서는 그의 연구에 대해 간략하게 소개하고 있습니다. 각 질문을 다루기 위한 메커니즘 혹은 개념이 무엇이고, 그를 다루기 위한 방법 혹은 도구를 소개합니다.



## 2. Causal Hierarchy

Causal Hierarchy는 위에서 말한 질문들에 대해 개념화하고 각 개념들의 포함관계를 소개하고 있습니다.

![/assets/img/posts/2021-02-21-The-Seven-Tools-of-Causal-Inference-for-Machine-Learning/Untitled.png](/assets/img/posts/2021-02-21-The-Seven-Tools-of-Causal-Inference-for-Machine-Learning/Untitled.png)

![/assets/img/posts/2021-02-21-The-Seven-Tools-of-Causal-Inference-for-Machine-Learning/Untitled_Diagram.png](/assets/img/posts/2021-02-21-The-Seven-Tools-of-Causal-Inference-for-Machine-Learning/Untitled_Diagram.png){: width="50%" height="50%"}



#### 1. Association



Association은 
$$
p(y|x)
$$
를 의미하는 것으로, 오로지 통계적 관계(statistical relationships)을 의미합니다. 예로, 치약을 구입하는 고객을 관찰하면 치실을 구입할 가능성이 높다 등은 Association에 해당하며 Conditional expectation으로 추론이 가능합니다. 여기서는 어떤 causal 정보도 필요하지 않으므로 가장 하위 개념이며, 현재 많은 머신러닝 방법들이 사용하는 방식입니다.



Association은 
$$
P(y|x)
$$
로 나타내며, $$X=x$$라는 사건을 관측했을 때, $$Y=y$$라는 사건이 나타날 확률을 의미합니다.



#### 2. Intervention



Intervention은 실제 관찰한 것을 바꾼 것으로, "What if I make it happen?"이라는 미래적 질문으로 대표될 수 있습니다. 예로, 가격을 두 배로 올리면 어떻게 될까요? 라는 질문에는 새 가격에 대한 고객의 선택 혹은 반응이 바뀌게 되므로 일반적으로 우리의 관측치들로는 답할 수 없습니다. 따라서 이는 Association보다 상위개념입니다.



Intervention은 
$$
P(y|do(x),z)
$$
와 같이 나타내며, $$Z=z$$라는 사건을 관측하고 Intervention을 통해 $$X=x$$라고 값을 부여했을 때(이를 intervention을 가했다는 의미에서 $$do(x)$$로 표기합니다), $$Y=y$$라는 사건이 발생할 확률을 의미합니다.



#### 3. Counterfactuals



Counterfactuals는 "What if I had acted differenctly"라는 질문으로, 회고적인 추론이 필요합니다. 이는 Intervention의 미래적 질문에 대해서도 답할 수 있으므로 3 가지 개념 중 가장 상위 개념입니다. 예로, 가격을 두 배로 하면 어떻게 될까요? 라는 질문은 가격이 두 배 였다면 어떻게 될까요? 라는 질문으로 포함될 수 있습니다. Counterfactuals은 약물 치료를 받은 피실험자들에 대해 실험을 재 시행할 수 없으므로, 약물을 받지 않았더라면 그들이 어떻게 행동했을까? 에 대한 질문을 다루고자 합니다. 혹은 민사 법원에서는 피고가 상해의 범인으로 간주되나, 피고의 조치에 대해선 상해가 발생하지 않았을 가능성이 더 높은데, 실제로 피고의 조치가 취해지지 않은 대안 세계를 비교하고자 하는 경우를 다룹니다.



Counterfactual은 
$$
P(y_x|x',y')
$$
와 같이 나타내며, 실제로 $$X=x',Y=y'$$라고 관찰했을 때, $$X=x$$라고 관측했었을 때 $$Y=y$$라는 사건이 나타났을 확률을 의미합니다.





## 3. Causal Inference

(to be updated...)







**Reference**

1. Pearl, J. (2019). The seven tools of causal inference, with reflections on machine learning. Communications of the ACM, 62(3), 54-60.