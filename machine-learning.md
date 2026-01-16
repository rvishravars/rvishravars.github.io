---
layout: page
title: Machine Learning
permalink: /machine-learning/
---

# Machine Learning

[← Back to Home](/)

Below is a collection of core ML terms and their overly simplistic one line explanations.

## Six steps in Machine learning

| Term | One-Line Explanation |
|-----|----------------------|
| Get Data | Collect and prepare the raw data needed for learning. |
| Space of Possible Solutions | Define the set of models or hypotheses the system is allowed to choose from. |
| Characterise Objective | Specify what the model should optimize, such as minimizing error or maximizing accuracy. |
| Find Algorithm | Select a learning method to search the solution space effectively. |
| Run | Execute the algorithm on the data to learn model parameters. |
| Validate | Evaluate performance on unseen data to assess generalization. |

## Building blocks

| Term | One-Line Explanation |
|-----|----------------------|
| Hypothesis | \( f_\theta \), a candidate function from the hypothesis class chosen to model the data. |
| Parameter | \( \theta \), the values (e.g., weights and biases) learned from data during training. |
| Hyperparameter | Settings like learning rate, regularization strength, or number of hidden units that are set **before training** and not learned from data. |
| Loss Function | \( L(y, \hat{y}) \), a measure of disagreement between true labels and predictions. |
| Train-Test Split | Dividing the dataset into training and test sets, e.g., 80%-20%, to train the model and evaluate its generalization on unseen data. |
| Training Error | \( \frac{1}{n} \sum_{i=1}^{n} L(f(x_i), y_i) \), the average loss on the training dataset. |
| Test Error | \( \frac{1}{m} \sum_{j=1}^{m} L(f(x_j^{test}), y_j^{test}) \), the average loss on unseen test data. |
| Cross-Validation Error | The average error computed by training and validating on multiple splits of the data to estimate generalization. |
| Best Straight Line | In linear regression, the line \( y = w^\top x + b \) that minimizes training loss (e.g., MSE) across the dataset. |
| Overfitting | When a model fits the training data too closely, capturing noise and performing poorly on unseen data. |
| Underfitting | When a model is too simple to capture the underlying pattern, leading to high error on both training and test data. |

## 🧠 Classification
| Term | One-Line Explanation |
|-----|----------------------|
| Hypothesis Class (Classification) | \( \mathcal{H} = \{ f_\theta : \mathcal{X} \rightarrow \{1,\dots,K\} \mid \theta \in \Theta \} \), the set of all classifiers mapping inputs to discrete class labels. |
| Hypothesis | \( f_\theta \), a candidate function from the hypothesis class chosen to model the data. |
| Parameter | \( \theta \), the values (e.g., weights and biases) learned from data during training. |
| Feature Representation | \( x \in \mathcal{X} \), the vector of input features that encodes the raw data for the learning algorithm. |
| Feature Transform | \( \phi(x) : \mathcal{X} \rightarrow \mathcal{F} \), a mapping that converts input features into a new space to make patterns easier to learn. |
| Feature Transform Example | \( \phi(x_1, x_2) = (x_1, x_2, x_1^2, x_2^2, x_1 x_2) \), lifting 2D inputs into a higher-dimensional space to make them linearly separable. |
| One-Hot Encoding | Converts categorical values into binary vectors, e.g., Red → [1,0,0], Green → [0,1,0], Blue → [0,0,1]. |
| Feature Standardization | \( x' = \frac{x - \mu}{\sigma} \), rescales features to have zero mean and unit variance. |
| Loss Function | \( L(y, \hat{y}) = -\sum_{k=1}^{K} \mathbf{1}[y = k] \log p_\theta(y = k \mid x) \), cross-entropy measuring disagreement between true and predicted probabilities. |
| Training Set | \( \frac{1}{n} \sum_{i=1}^{n} L(f(x_i), y_i) \), the average loss on the training data. |
| Test Set Error | \( \frac{1}{m} \sum_{j=1}^{m} L(f(x_j^{test}), y_j^{test}) \), the average loss on unseen test data. |
| ML as Optimization | \( \theta^* = \arg\min_{\theta \in \Theta} \frac{1}{n} \sum_{i=1}^{n} L(f_\theta(x_i), y_i) \), minimizing empirical risk. |
| Linear Classifier (Sign Function) | \( f(x) = \operatorname{sign}(w^\top x + b) \), assigns class based on which side of the hyperplane the input lies. |
| Types of Linear Classifiers | Perceptron, Logistic Regression, Linear SVM, Least-Squares Classifier, all using linear decision boundaries. |
| Linear Separability | A dataset is linearly separable if a hyperplane exists that perfectly separates classes. |
| Perceptron | A binary linear classifier: \( f(x) = \operatorname{sign}(w^\top x + b) \). |
| Perceptron Through Origin | Perceptron with no bias term (\( b = 0 \)), hyperplane passes through origin. |
| Perceptron Intuition | Adjust decision boundary to correctly classify misclassified points. |
| Perceptron Algorithm | Iteratively updates \( w \) and \( b \) using misclassified points until convergence. |
| Margin of Data Point | Perpendicular distance from a point to the decision boundary: \( \gamma_i = \frac{y_i (w^\top x_i + b)}{\|w\|} \). |
| Margin of Dataset | Smallest margin among all points: \( \gamma = \min_i \gamma_i \). |
| Perceptron Convergence | If dataset is linearly separable with margin \( \gamma \), the algorithm converges in at most \( \frac{R^2}{\gamma^2} \) updates, \( R = \max_i \|x_i\| \). |
| Logistic Regression | Linear classifier estimating class probabilities using sigmoid function. |
| Logistic Regression Hypothesis | \( h_\theta(x) = \sigma(w^\top x + b) = \frac{1}{1 + e^{-(w^\top x + b)}} \), models \( P(y=1 \mid x) \). |
| Sigmoid Function | \( \sigma(z) = \frac{1}{1 + e^{-z}} \), maps real-valued inputs to probabilities in (0,1). |
| Sigmoid Characteristics | S-shaped, monotonic, differentiable, vanishing gradients for large \(|z|\). |
| Sigmoid Function (Visualization) | Plot of \( \sigma(z) \), showing smooth S-shaped transition from 0 to 1. |
| Cross-Entropy Loss | \( L(y,\hat{y}) = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})] \), penalizes confident wrong predictions. |
| Sigmoid and Probability | \( \sigma(w^\top x + b) = P(y=1 \mid x) \); log converts products into sums for convex, stable optimization. |
| Gradient Descent (Logistic Regression) | \( \theta \leftarrow \theta - \eta \nabla_\theta \frac{1}{n}\sum_i L(y_i, \sigma(w^\top x_i + b)) \), updates parameters to minimize loss. |
| Logistic Regression Regularization | \( L(\theta) = -\frac{1}{n} [y^\top \log(\sigma(X\theta)) + (1-y)^\top \log(1-\sigma(X\theta))] + \lambda \|\theta\|^2 \), reduces overfitting. |
| Regularization Parameter (\( \lambda \)) | Controls regularization strength, balancing fit and complexity. |
| Regularization Uses | Prevents overfitting, improves generalization, stabilizes parameters, can enforce sparsity. |
| Important Hyperparameters (Logistic Regression) | Learning rate \( \eta \), number of iterations, feature dimensions, regularization constant \( \lambda \). |

## 🧩 Clustering

| **Term** | **One-Line Explanation** |
|-----------|---------------------------|
| **Clustering** | Grouping data points so that those within a cluster are more similar to each other than to those in other clusters. |
| **K-means** | A partitioning algorithm that iteratively assigns points to the nearest cluster mean and updates the means until convergence. |
| **k-means++** | A smart initialization method for k-means that spreads out initial cluster centers for better results. |
| **Fuzzy c-means** | A clustering method allowing data points to have fractional membership in multiple clusters. |
| **Expectation-Maximization (EM)** | Alternates between assigning data points to clusters (E-step) and updating cluster parameters (M-step) to maximize likelihood. |
| **Gaussian Mixture Model (GMM)** | Represents data as a mixture of multiple Gaussian distributions learned via EM. |
| **DBSCAN** | A density-based clustering algorithm that groups closely packed points and marks outliers as noise. |
| **Core Point** | A point with at least *MinPts* neighbors within distance *Eps*, forming the dense core of a cluster. |
| **Hierarchical Clustering** | Builds nested clusters using either a bottom-up (agglomerative) or top-down (divisive) strategy. |
| **Agglomerative Clustering** | Starts with each point as its own cluster and merges them iteratively based on distance. |

## 🔁 Online Clustering

| **Term** | **One-Line Explanation** |
|-----------|---------------------------|
| **Online Learning** | Processes data incrementally as it arrives, adapting models continuously. |
| **Online Averaging** | Updates the mean iteratively with each new data point using a learning rate γₙ. |
| **Competitive Learning (CL)** | Neurons compete to represent input data; only the winner updates its weight. |
| **Self-Organizing Map (SOM)** | Adds a topological structure so nearby neurons update together, forming organized feature maps. |
| **Neural Gas** | Removes SOM’s grid topology and ranks neurons by distance to ensure fair updates. |
| **Leader-Follower Algorithm** | Creates new clusters when incoming data exceed a distance threshold from all existing cluster centers. |
| **Change Detection** | Identifies anomalies when new data fall outside a statistical “three-sigma” range. |

## 📉 Dimension Reduction

| **Term** | **One-Line Explanation** |
|-----------|---------------------------|
| **Principal Component Analysis (PCA)** | Projects data to directions (principal components) that maximize variance. |
| **Eigenvector** | The direction of maximum variance in PCA, representing a principal component. |
| **Eigenvalue** | Measures how much variance is captured by each eigenvector. |
| **Hebbian Learning** | Neural learning principle stating that “neurons that fire together wire together.” |
| **Oja’s Rule** | An online learning rule for neural PCA using Hebbian learning. |
| **Kernel Trick** | Computes dot products in a high-dimensional “feature” space without explicitly performing the transformation. |
| **Kernel PCA** | Extends PCA into a nonlinear feature space using kernel functions. |
| **Multi-Dimensional Scaling (MDS)** | Reduces dimensions by preserving pairwise distances between points. |
| **Sammon’s Mapping** | A nonlinear MDS method that minimizes distance distortion between original and reduced spaces. |
| **Isomap** | Uses graph geodesic distances to preserve manifold structure during dimension reduction. |

## 🧠 Advanced Classification

| **Term** | **One-Line Explanation** |
|-----------|---------------------------|
| **Supervised Learning** | Uses labeled data to train a model that predicts outcomes for unseen data. |
| **Unsupervised Learning** | Finds patterns in unlabeled data such as clusters or latent features. |
| **Bayes’ Theorem** | Updates probability estimates based on new evidence. |
| **Naïve Bayes Classifier** | Assumes feature independence and applies Bayes’ theorem for fast classification. |
| **k-Nearest Neighbour (k-NN)** | Classifies data points based on the majority class among their k nearest neighbors. |
| **Linear Discriminant Analysis (LDA)** | Projects data to maximize class separability using linear combinations of features. |
| **Learning Vector Quantization (LVQ)** | A supervised version of vector quantization that tunes prototypes using labeled data. |
| **Radial Basis Function (RBF) Network** | A neural model using Gaussian functions as hidden neurons to interpolate or classify data. |
| **Support Vector Machine (SVM)** | Finds an optimal hyperplane that maximizes the margin between classes. |
| **Decision Tree** | A tree-structured model that splits data recursively to reduce uncertainty or impurity. |
| **Entropy** | A measure of uncertainty or impurity in a dataset. |
| **Gini Index** | A simpler impurity measure used in decision tree algorithms like CART. |
| **Feature Importance** | Quantifies each feature’s contribution to reducing impurity in a decision tree. |

## Feature Selection

| **Term** | **One-line Explanation** |
|-----------|---------------------------|
| Feature Selection | Process of choosing a subset of relevant features to improve model performance and interpretability. |
| Filter Method | Ranks features using statistical measures like correlation or mutual information. |
| Wrapper Method | Uses a learning algorithm to evaluate and select feature subsets through search strategies. |
| Embedded Method | Performs feature selection during model training (e.g., LASSO, decision trees). |
| Pearson Correlation | Measures linear dependence between a feature and the target variable. |
| Mutual Information | Quantifies how much a feature reduces uncertainty about the target variable. |
| Boruta Algorithm | Random forest-based method comparing real vs. shuffled “shadow” features. |
| SHAP Value | Game-theoretic approach to measure feature importance via contribution to predictions. |
| mRMR | Selects features that are maximally relevant to the target but minimally redundant. |
| Sequential Forward Selection | Adds one feature at a time that improves performance most. |
| Sequential Backward Elimination | Starts with all features, removes least useful features iteratively. |

---

## Performance Evaluation

| **Term** | **One-line Explanation** |
|-----------|---------------------------|
| Confusion Matrix | Table summarizing true vs predicted classifications. |
| Accuracy | Proportion of correctly classified samples. |
| Precision | Fraction of positive predictions that are correct. |
| Recall / Sensitivity | Fraction of true positives correctly identified. |
| Specificity | Fraction of true negatives correctly identified. |
| F1 Score | Harmonic mean of precision and recall. |
| MCC | Balanced metric for binary classification, effective for imbalanced data. |
| ROC Curve | Plots true positive rate vs. false positive rate at varying thresholds. |
| AUC | Area under the ROC curve; measures overall model discrimination ability. |
| Cross-Validation | Evaluates model performance using multiple data splits. |

---

## Regression

| **Term** | **One-line Explanation** |
|-----------|---------------------------|
| Regression | Predicts continuous numeric outcomes from input features. |
| Ordinary Least Squares | Minimizes mean squared error to estimate model coefficients. |
| In-sample Error | Error on the training data used for fitting the model. |
| Out-of-sample Error | Error on unseen test data, measuring generalization. |
| Linear Regression | Predicts outcome as a weighted sum of input features. |
| Polynomial Regression | Extends linear regression by including polynomial terms of inputs. |
| Gradient Descent | Iterative optimization method for minimizing loss. |
| Learning Rate | Step size determining how much to adjust parameters in gradient descent. |
| Regularization | Penalizes model complexity to prevent overfitting. |
| Ridge Regression | L2 regularization that shrinks coefficients to reduce variance. |
| LASSO Regression | L1 regularization that drives some coefficients to zero (feature selection). |
| Decision Tree Regressor | Splits data space recursively to minimize prediction error. |
| R² (Coefficient of Determination) | Measures proportion of variance explained by the model. |
| RMSE | Square root of mean squared error; penalizes large errors. |
| MAE | Mean of absolute prediction errors; less sensitive to outliers. |
| Regularization Parameter (λ) | Controls trade-off between bias and variance. |

---

## Ensemble Learning

| **Term** | **One-line Explanation** |
|-----------|---------------------------|
| Ensemble Learning | Combines multiple models to improve overall prediction accuracy. |
| Bagging | Trains models on bootstrapped samples and averages predictions. |
| Boosting | Sequentially trains models to focus on previous errors. |
| Random Forest | Ensemble of decision trees using random feature and data sampling. |
| AdaBoost | Assigns higher weights to misclassified samples for next learner. |
| Gradient Boosting | Fits new models to residuals from previous models. |
| Voting | Combines predictions via majority or weighted averaging. |
| Diversity | Independence among base learners that improves ensemble performance. |

---

## Model Selection

| **Term** | **One-line Explanation** |
|-----------|---------------------------|
| Bias | Systematic error due to overly simple models. |
| Variance | Sensitivity to fluctuations in training data. |
| Bias-Variance Tradeoff | Balancing underfitting (bias) vs overfitting (variance). |
| Cross-Validation | Estimates generalization by testing on unseen folds. |
| Regularization | Adds penalty terms to control model complexity. |
| AIC | Akaike Information Criterion for model comparison using likelihood and parameters. |
| BIC | Bayesian Information Criterion penalizing model complexity more strongly. |
| Hyperparameter Tuning | Adjusting non-learnable settings (e.g., k in KNN) to optimize performance. |

---

## Multilayer Perceptron

| **Term** | **One-line Explanation** |
|-----------|---------------------------|
| Perceptron | Basic linear classifier using weighted sums and activation. |
| Activation Function | Nonlinear function applied to neuron outputs (e.g., ReLU, sigmoid). |
| Backpropagation | Algorithm to compute gradients and update weights in neural networks. |
| Learning Rate | Determines how fast weights are updated during training. |
| Momentum | Adds a fraction of previous weight update to speed convergence. |
| Vanishing Gradient | Gradients shrink in deep networks, hindering learning. |
| Mini-batch Gradient Descent | Uses small data subsets for faster, smoother updates. |
| ADAM | Adaptive optimization combining momentum and RMSprop ideas. |

---

## Convolutional Neural Networks

| **Term** | **One-line Explanation** |
|-----------|---------------------------|
| Convolution | Operation applying filters over input data to extract features. |
| Pooling | Reduces spatial size while retaining important information. |
| LeNet-5 | Early CNN architecture for digit recognition. |
| AlexNet | Deep CNN that popularized GPU training and ReLU activations. |
| YOLO | Real-time object detection architecture (“You Only Look Once”). |
| ResNet | Deep CNN using residual (skip) connections to ease training. |
| Vision Transformer | Applies attention mechanisms from NLP to image recognition. |

---

## Variational Autoencoder (VAE)

| **Term** | **One-line Explanation** |
|-----------|---------------------------|
| Autoencoder | Neural network that reconstructs inputs from compressed latent space. |
| Variational Autoencoder | Probabilistic autoencoder learning a smooth latent distribution. |
| Encoder | Maps input data to latent variables. |
| Decoder | Reconstructs data from latent variables. |
| Latent Space | Compressed representation capturing essential features. |
| KL Divergence | Measures difference between two probability distributions. |
| Reparameterization Trick | Enables gradient-based learning through stochastic sampling. |
| ELBO | Evidence Lower Bound, objective function optimized in VAEs. |

---

## Generative Adversarial Network (GAN)

| **Term** | **One-line Explanation** |
|-----------|---------------------------|
| GAN | Framework with generator and discriminator competing in a minimax game. |
| Generator | Produces synthetic data from random noise. |
| Discriminator | Distinguishes real data from generated data. |
| Minimax Game | Optimization framework where G tries to fool D and D tries to detect fakes. |
| Mode Collapse | Failure where generator produces limited data diversity. |
| Nash Equilibrium | State where neither G nor D can improve performance unilaterally. |
| CycleGAN | Uses cyclic consistency for image-to-image translation without paired data. |
| Super-Resolution GAN | Enhances image resolution using adversarial and perceptual loss. |

---

## Transfer Learning

| **Term** | **One-line Explanation** |
|-----------|--------------------------|
| **Transfer Learning (TL)** | Applying knowledge learned in one domain/task to another related domain/task. |
| **Source Domain** | The domain providing prior knowledge or training data. |
| **Target Domain** | The new domain where learned knowledge is applied. |
| **Inductive Transfer Learning** | TL where labeled data exists in the target domain. |
| **Transductive Transfer Learning** | TL with labeled data only in the source domain. |
| **Unsupervised Transfer Learning** | TL where both domains lack labeled data. |
| **Self-taught Learning** | Learning features from unlabeled data and transferring them to labeled tasks. |
| **Domain Adaptation** | Adapting models from a source domain to an unlabeled target domain. |
| **Instance-based Transfer** | Re-weighting instances from the source domain to reduce domain bias. |
| **TrAdaBoost** | Boosting-based TL algorithm that adjusts instance weights between domains. |
| **Sparse Coding** | Learning sparse feature representations to transfer across domains. |
| **Hierarchical Transfer** | Multi-level knowledge transfer to handle complex or multi-task learning. |
| **Covariate Shift** | Situation where input distributions differ between training and testing domains. |

---

## Deep Transfer Learning

| **Term** | **One-line Explanation** |
|-----------|--------------------------|
| **Deep Transfer Learning (DTL)** | Using deep neural networks to transfer features or representations between domains. |
| **Domain Invariance** | Ensuring learned features are independent of domain differences. |
| **Adversarial Domain Adaptation (DANN)** | Using adversarial training to make feature representations domain-invariant. |
| **VAE (Variational Autoencoder)** | A generative model used to learn continuous latent representations. |
| **GAN (Generative Adversarial Network)** | A generative model where two networks compete to produce realistic outputs. |
| **UNIT (Unsupervised Image-to-Image Translation)** | Framework combining VAE and GAN to translate images across domains. |
| **CGGS/DATL** | Cross-Grafted Generative Stacks / Deep Adversarial Transfer Learning; creates transition domains. |
| **Latent Space Mixup** | Mixing latent representations to create hybrid samples for domain generalization. |
| **CDLM (Cross-Domain Latent Modulation)** | A method to modulate latent variables for bi-directional domain adaptation. |
| **Taskonomy** | Study of task relationships to learn how tasks transfer knowledge efficiently. |
| **t-SNE Visualization** | A technique for visualizing high-dimensional domain representations. |
| **Affinity Matrix** | Measures transferability among tasks based on learned representations. |

---

## Federated Learning

| **Term** | **One-line Explanation** |
|-----------|--------------------------|
| **Federated Learning (FL)** | Collaborative training of models across multiple devices without sharing raw data. |
| **FedAvg** | A core FL algorithm that averages locally trained model weights across clients. |
| **Split Learning** | Dividing models between clients and servers to preserve data privacy. |
| **Data Heterogeneity** | Variation in local datasets across clients (non-IID data). |
| **Resource Heterogeneity** | Differences in client computational and communication capabilities. |
| **Hetero-FL** | Federated setup allowing clients to use different model architectures. |
| **FedMD** | FL framework using model distillation and transfer learning for heterogeneous clients. |
| **CloREF** | Rule-based collaborative FL method allowing different model types (e.g., RF, SVM). |
| **Gradient Inversion Attack** | Technique to reconstruct private data from shared gradients. |
| **Secure Aggregation** | Cryptographic protocol to protect individual client updates during aggregation. |
| **LOKI Attack** | Advanced data leakage attack that reconstructs data even in secure aggregation. |
| **Flower Framework** | Open-source framework for building and simulating federated learning systems. |

---

## References

- Pan & Yang, *A Survey on Transfer Learning*, IEEE TKDE 2010.
- Raina et al., *Self-taught Learning*, ICML 2007.
- Ganin & Lempitsky, *Unsupervised Domain Adaptation by Backpropagation (DANN)*, ICML 2015.
- Liu et al., *UNIT: Unsupervised Image-to-Image Translation Networks*, NeurIPS 2017.
- Xu et al., *Adversarial Domain Adaptation with Domain Mixup*, AAAI 2020.
- Hou et al., *Cross-Domain Latent Modulation*, WACV 2021.
- McMahan et al., *Communication-efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017.
- Bonawitz et al., *Practical Secure Aggregation for Federated Learning*, NIPS 2016.
- Pang et al., *Rule-based Collaborative Learning (CloREF)*, PAKDD 2022 / IEEE TKDE 2023.

---

[↑ Back to Top](#machine-learning)
