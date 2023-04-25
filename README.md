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

### Sec. III. Algorithm Implementation and Development

### Sec. IV. Computational Results

### Sec. V. Summary and Conclusions
