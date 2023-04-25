# MNIST-Classification-ML

### Table of Contents
[Abstract](#Abstract)
<a name="Abstract"/>

[Sec. I. Introduction and Overview](#sec-i-introduction-and-overview)     
<a name="sec-i-introduction-and-overview"/>

[Sec. II. Theoretical Background](#sec-ii-theoretical-background)     
<a name="sec-ii-theoretical-background"/>

[Sec. III. Algorithm Implementation and Development](#sec-iii-algorithm-implementation-and-development)
<a name="sec-iii-algorithm-implementation-and-development"/>

[Sec. IV. Computational Results](#sec-iv-computational-results)
<a name="sec-iv-computational-results"/>

[Sec. V. Summary and Conclusions](#sec-v-summary-and-conclusions)
<a name="sec-v-summary-and-conclusions"/>


### Abstract

The MNIST dataset is a well-known dataset of handwritten digits that is often used in machine learning research. In this project, the goal is to perform an analysis of the MNIST dataset, including an SVD analysis of the digit images, building classifiers to identify individual digits in the training set, and comparing the performance of LDA, SVM, and decision tree classifiers on the hardest and easiest pair of digits to separate.

### Sec. I. Introduction and Overview
#### Introduction:

The MNIST dataset is a large collection of handwritten digit images that has become a standard benchmark for testing machine learning algorithms. In this analysis, we will perform an in-depth analysis of the MNIST dataset, including an SVD analysis of the digit images, building a classifier to identify individual digits in the training set, and comparing the performance of different classification algorithms.

#### Overview:

We will start by performing an SVD analysis of the digit images, which involves reshaping each image into a column vector and using each column of the data matrix as a different image. We will analyze the singular value spectrum and determine how many modes are necessary for good image reconstruction (i.e. the rank r of the digit space). We will also interpret the U, Σ, and V matrices in this context.

Next, we will project the data into PCA space and build a classifier to identify individual digits in the training set. We will start by picking two digits and building a linear classifier (LDA) that can reasonably identify/classify them. Then we will pick three digits and try to build a linear classifier to identify these three. We will determine which two digits in the data set appear to be the most difficult to separate and quantify the accuracy of the separation with LDA on the test data. We will also determine which two digits in the data set are most easy to separate and quantify the accuracy of the separation with LDA on the test data.

Finally, we will compare the performance between LDA, SVM, and decision trees on the hardest and easiest pair of digits to separate (as determined in the previous step). SVM and decision tree classifiers were the state-of-the-art classification techniques until about 2014, and we will see how well they separate between all ten digits in the MNIST dataset.

###  Sec. II. Theoretical Background

The **MNIST dataset** is a popular dataset for image classification and machine learning tasks. It consists of 70,000 handwritten digits, with 60,000 images in the training set and 10,000 images in the test set. Each image is a grayscale 28x28 pixel image, representing a digit from 0 to 9.

**PCA (Principal Component Analysis)** is a technique used to reduce the dimensionality of data while preserving the most important information in the data. It finds the principal components by performing a linear transformation of the data into a new coordinate system, where the axes are orthogonal (perpendicular) and ranked by the amount of variance they explain in the data. The first principal component is the direction of maximum variance, the second principal component is the direction of maximum variance that is orthogonal to the first, and so on.

Mathematically, PCA finds the principal components by computing the eigenvectors and eigenvalues of the covariance matrix of the data. The eigenvectors represent the directions of the principal components, while the eigenvalues represent the amount of variance explained by each principal component. The eigenvectors are sorted by their corresponding eigenvalues in descending order, so the first eigenvector corresponds to the first principal component.

PCA can be used for various purposes such as data compression, data visualization, and feature extraction. In the context of image analysis, PCA can be used to identify the most significant features in images, and to project images onto a lower-dimensional space for further analysis or visualization.

**SVD (Singular Value Decomposition)** is a more general decomposition method that can be applied to any matrix, unlike eigendecomposition, which is only applicable to square matrices. In this project, we will use SVD to visualize the singular value spectrum and use this to find the rank _r_ of the digit space.

SVD is a matrix factorization technique that decomposes a matrix M (m*n) into three matrices: U, Σ, and V* (_in the code, these matricies have respectively been called U, s, and V_). The matrix Σ is diagonal and contains the singular values (in decreasing order) of the original matrix, which correspond to the square roots of the eigenvalues of the covariance matrix of the data. The matrix U contains the left singular vectors, and the matrix V* contains the right singular vectors.

It is important to note that any time a matrix A is multiplied by another matrix B, only two primary things can occur: B will _stretch_ and _rotate_ A. Hence, the three matricies SVD splits a matrix M into simply rotate M (V*), stretch M (Σ), and rotate M (U).

This concept can be visualized in figure 1 below. If we have a 2D disk (of a real sqaure matrix M), rotation first occurs, then stretching M will create an elipse with major and minor axes σ1 and σ2, and then we rotate again and find ourselves in a new coordiante system. We can say that u1 and u2 are unit orthonormal vectors known as principal axes along σ1 and σ2 and σ1 and σ2 are singular values. Thus, u1 and u2 determine the direction in which the stretching occurs while the singular valeus determine the magnitude of this stretch (eg. σ1u1).

![image](https://user-images.githubusercontent.com/116219100/232977674-cfd2a2f3-18ae-41e9-93cf-9585034cc857.png)
*Figure 1: Visualization of the 3 SVD matrices U, Σ, and V*

In the context of PCA, the principal components can be obtained from the right singular vectors of the data matrix. This is because the right singular vectors are the eigenvectors of the covariance matrix of the data. Therefore, the right singular vectors are often called the PCA modes.

However, the left singular vectors of the data matrix can also be useful in some applications, such as in image compression. In this case, the left singular vectors are called the SVD modes. The SVD modes and the PCA modes are related, but they are not exactly the same thing, as the SVD modes are not guaranteed to be orthogonal, while the PCA modes are always orthogonal.

LDA (Linear Discriminant Analysis), SVM (Support Vector Machines), and decision tree classifiers are **machine learning algorithms used for classification tasks**.

**LDA (Linear Discriminant Analysis)** is a linear classification method that finds a linear combination of features that best separates the classes in the data. It assumes that the data is normally distributed, and the classes have the same covariance matrix. LDA can be used for binary and multiclass classification tasks.

**SVM (Support Vector Machines)** is a non-linear classification method that finds a hyperplane that separates the classes in the data with the largest margin. It uses a kernel function to map the data into a higher-dimensional space, where the classes are linearly separable. SVM can be used for binary and multiclass classification tasks, and is known for its ability to handle high-dimensional data.

**Decision tree classifiers** are a type of algorithm that uses a tree-like model of decisions and their possible consequences to classify the data. It works by splitting the data into subsets based on the values of one feature at a time, until the subsets are as pure as possible (i.e., they only contain data points from one class). The tree structure can be used to interpret the classification decisions made by the algorithm.

All three classifiers have their own advantages and disadvantages, and their performance can vary depending on the specific data and task at hand. LDA and SVM are commonly used for classification tasks in a wide range of fields, while decision trees are often used in applications where interpretability is important.

### Sec. III. Algorithm Implementation and Development

Firstly, I loaded the MNIST dataset from the scikit-learn library using the fetch_openml function:

```
from sklearn.datasets import fetch_openml

# Load the MNIST dataset
mnist = fetch_openml('mnist_784')

# Reshape each image into a column vector
X = mnist.data.T / 255.0  # Scale the data to [0, 1]
Y = mnist.target
```

The MNIST dataset is a 2D array where each row corresponds to an image and each column corresponds to a pixel value. This code uses the transpose (.T) method to convert the dataset into a 784x70000 array where each column corresponds to an image and each row corresponds to a pixel value. This is often a more convenient format for performing computations and analyses.

Additionally, the pixel values are divided by 255 to scale them down to a range of [0,1]. This is because the original pixel values range from 0-255, but many machine learning algorithms work better when the data is scaled to be between 0 and 1.

Also, this code creates an array Y containing the target labels for each image in the dataset. These labels indicate which digit (0-9) is shown in each image, and will be used as the target variable for classification tasks.

We then run SVD and store the three matricies formed in variables U, s, and V:

```
# Perform SVD on X
U, s, V = np.linalg.svd(X, full_matrices=False)
```

As discussed in the theoretical background section, we can use the s (Σ) matrix to obtain the singular values in descending order. Hence, we can simply plot s to get the singular value spectrum like this:

```
# Plot the singular value spectrum
plt.plot(s)
plt.xlabel('Singular Value Index')
plt.ylabel('Singular Value')
plt.show()
```

The singular value spectrum from SVD represents the singular values of the original matrix. In the context of image reconstruction, it represents the amount of energy captured by each mode (or singular value) of the SVD. The singular values are arranged in decreasing order, so the first singular value captures the most energy and the last singular value captures the least.

To determine the number of modes necessary for good image reconstruction, we need to look at the amount of energy captured by each mode. One way to do this is to calculate the cumulative sum of the singular values and normalize by the total sum of all singular values. Then we can plot this cumulative sum and look for an "elbow" or "knee" point where adding more modes does not contribute much to the total energy captured.

The rank r of the digit space can be determined by looking at the number of non-zero singular values in the singular value spectrum. This is because the rank of the original matrix is equal to the number of non-zero singular values. We use the following snippet of code to plot the cumulative sum and then find _r_:

```
# Plot the cumulative sum of the singular values
plt.plot(np.cumsum(s)/np.sum(s))
plt.xlabel('Singular Value Index')
plt.ylabel('Cumulative Sum')
plt.show()

# Set the threshold for the amount of variance to retain
threshold = 0.46

# Calculate the number of dimensions to keep
k = np.argmax(np.cumsum(s) >= threshold*np.sum(s)) + 1

# Calculate the number of singular values greater than s[k-1] (where k is the number of dimensions to keep)
r = np.sum(s > s[k-1])

# Print the number of dimensions and the number of retained singular values
print('k:', k)
print('r:', r)
```

Here, the code first sets a threshold for the amount of variance to retain from the SVD, which is a value between 0 and 1. It then uses the cumulative sum of the singular values (stored in s) to find the index (k) of the last singular value that, when added to the sum of the preceding singular values, exceeds the threshold times the total sum of the singular values. The number of dimensions to keep is then k. Finally, the number of singular values greater than s[k-1] is counted to obtain the number of singular values that are retained, which is stored in r.

The results will be analyzed in depth in Section IV. 
### Sec. IV. Computational Results

### Sec. V. Summary and Conclusions
