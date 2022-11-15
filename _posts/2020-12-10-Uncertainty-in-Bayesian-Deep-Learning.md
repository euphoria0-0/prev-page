---
title: Uncertainty in Bayesian Deep Learning
author: euphoria0-0
date: 2020-12-10 23:30:00 +0800
categories: [AI, Paper Review]
tags: [Machine Learning, Bayesian, Uncertainty, Computer Vision, Paper Review]
toc: true
math: true
comments: true
---



이번 글에서는 Yarin Gal (및 그 외)이 쏘아올린 공.. 바로 Uncertainty에 대하여 소개하고자 합니다. 이를 소개하기 위해 그의 팀에서 나온 논문 몇가지를 함께 소개하고자 합니다.

논문 리스트이자, 이 글의 순서는 다음과 같습니다.

4. What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?
5. Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
3. Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning
4. Bayesian convolutional neural networks with Bernoulli approximate variational inference
5. Uncertainty in Deep Learning





## What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?

**Understanding what the model doesn't know!**

오늘날 Deep Learning 모델의 다양한 실패 사례들이 있습니다. 그중에서 특히, safety가 중요한 자율주행은 생명의 위험이 있을 수 있으며, ethical 문제를 야기할 수 있는 분야(예로, Google Photo 예시) 에는 사회적으로 문제가 제기되고 있습니다. 따라서 우리는 모델이 예측할 때, 모델이 잘 모르지만 어떻게든 결론을 내고 판단하는 것이 무엇인지 알아야 합니다. 그리고 그 잘 모르는 '불확실성'을 알 수 있어야 합니다. 우리는 이를 위해 'uncertainty'라는 개념을 가져옵니다.

**Uncertainty**

1. **Aleatoric Uncertainty** (data uncertainty)

   <kbd>Aleatoric Uncertainty</kbd>는 data에 대한 uncertainty로, 관측치에 내재된 노이즈에 대한 불확실성을 의미합니다. 이는 데이터가 포함하고 있는 다양한 noise를 내포하므로 데이터가 많다고 해서 줄일 수 없습니다. 예로 측정 오차 등이 포함될 수 있습니다.

   Aleatoric Uncertainty는 Homoscedastic uncertainty, Heteroscedastic uncertainty로 나뉠 수 있습니다. <kbd>Homoscedastic uncertainty</kbd>는 다른 input들에 대해서 constant한 불확실성입니다. input에 대해서는 같지만, 어떤 task에 대한 input이냐에 따라 달라질 수 있습니다. <kbd>Heteroscedastic uncertainty</kbd>는 모델에서 더 noisy한 output을 잠재적으로 가질 수 있는 input에 의존합니다. 따라서 이는 확실하지 않은 task에 대해 confident prediction을 주지 않도록 하기 위해 Computer Vision에서 더 중요합니다.

   Aleatoric Uncertainty는 데이터가 크다고 해도 줄어들지 않으므로 Large data의 상황에서 살펴볼 필요가 있습니다. 또한, (뒤에서 살펴보겠지만) Epistemic Uncertainty와 달리 계산 비용이 많이 필요없기 때문에 Real-time application에서 사용되기 쉽습니다.

2. **Epistemic Uncertainty** (model uncertainty)

   <kbd>Epistemic Uncertainty</kbd>는 model parameter에 대한 uncertainty로, 어떤 모델이 데이터를 생성했고 예측하였는지에 대한 무지(ignorance; 無知)에 기인한 불확실성을 의미합니다. 이는 분산의 추정치 형태로 나타나게 되므로 데이터가 많을수록 모델이 (epistemic에 한해) 추정을 잘 하게 되므로 Epistemic uncertainty는 Large data에 의해 설명될 수 있습니다.

   Epistemic Uncertainty는 training data와 다른 데이터를 이해할 때 도움이 됩니다. 왜냐하면, 이는 (표본) 분산의 형태로 추정되는데, 이 과정에서 이미 주어진 데이터들과 얼마나 다른지를 알 수 있기 때문입니다. 따라서, training data에 없는 새로운 데이터가 왔을 때 불확실성을 제대로 측정할 수 있어야 하므로 safety가 중요한 분야에서 더욱 중요합니다. 또한, large dataset에서는 충분히 줄어들 수 있으므로 small dataset에서 Epistemic Uncertainty를 보는 것이 중요합니다.

**for modeling two uncertainties**

기존에는 위에서 설명한 uncertainty에 대해 따로 학습하는 연구들만 존재해왔습니다. 하지만 이 논문을 통해 Aleatoric uncertainty와 Epistemic Uncertainty를 함께 학습할 수 있는 방법을 소개합니다.

 

























## Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics















**Reference**

[1] *Gal, Y., & Ghahramani, Z. (2016, June). Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning (pp. 1050-1059).*

[2] *Gal, Y., & Ghahramani, Z. (2015). Bayesian convolutional neural networks with Bernoulli approximate variational inference. arXiv preprint arXiv:1506.02158.*

[3] *Gal, Y. (2016). Uncertainty in deep learning. University of Cambridge, 1(3).*

[4] *Kendall, A., & Gal, Y. (2017). What uncertainties do we need in bayesian deep learning for computer vision?. In Advances in neural information processing systems (pp. 5574-5584).*

[5] *Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7482-7491).*



