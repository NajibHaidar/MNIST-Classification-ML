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

For a clearer view on when these singular values started to dip, I also plotted using a scree plot:

```
# calculate the cumulative sum of the squared singular values
cumulative_sum = np.cumsum(s**2)

# calculate the percentage of total variance captured by each singular value
variance_explained = cumulative_sum / np.sum(s**2)

# plot the scree plot
plt.plot(np.arange(1, len(s)+1), variance_explained, 'ro-', linewidth=2)
plt.axvline(x=10, color='b', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()
```

Then, the data was projected onto different V-modes, starting with 2,3, and 5:

```
# Select the V-modes to use
v_modes = [2, 3, 5]

# Project the data onto the selected V-modes
X_projected = V[:, v_modes].T @ X.values

# Create a 3D plot with a larger size
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot the projected data with a legend
scatter = ax.scatter(X_projected[0, :], X_projected[1, :], X_projected[2, :], c=Y.astype(int), cmap='jet')
legend = ax.legend(*scatter.legend_elements(), title='Classes', loc='upper left')
ax.add_artist(legend)

# Add axis labels
ax.set_xlabel(f'V-mode {v_modes[0]}')
ax.set_ylabel(f'V-mode {v_modes[1]}')
ax.set_zlabel(f'V-mode {v_modes[2]}')

ax.set_title('Projection onto V-modes 2, 3, and 5')

# # Set the view angle
# ax.view_init(90, 0)

# Show the plot
plt.tight_layout()
plt.show()
```

First, we select the three V-modes (columns of the V matrix) that we want to use for the projection. Next, we compute the projection of the data matrix X onto the selected V-modes, by multiplying the transpose of the selected V-modes with the data matrix. Note that we first transpose the selected V-modes so that they have the same shape as the data matrix. The rest simply plots the data into as a scatter plot. It should be noted here that the parameter _c_ in the scatter function represents the color of the markers in the scatter plot. It can take an array of values to color each marker differently based on some categorical or continuous variable. In this case, Y.astype(int) is passed as the argument for c, which is an array of the target values of the MNIST dataset, converted to integers. Therefore, each marker in the scatter plot will be colored based on its corresponding digit label. The cmap parameter is used to specify the colormap to use for coloring the markers. 'jet' is a popular choice of colormap that maps low values to blue, intermediate values to green and high values to red.

Now, LDA was performed to classify between 2 digits:

```
# LDA for 2 digits
# Perform PCA on the original dataset to reduce dimensionality to 10 components
pca = PCA(n_components=10)
pca.fit(X)
X_pca = pca.transform(X)

# Select only 4s and 9s in the new PCA space
X_pca_49 = X_pca[(Y == 4) | (Y == 9)]
Y_pca_49 = Y[(Y == 4) | (Y == 9)]

# Split into training and testing sets
X_pca_train, X_pca_test, Y_pca_train, Y_pca_test = train_test_split(X_pca_49, Y_pca_49, test_size=0.2, random_state=42)

# Train an LDA model on the training set using the transformed data
lda_pca = LinearDiscriminantAnalysis()
lda_pca.fit(X_pca_train, Y_pca_train)

# Make predictions on the testing set
Y_pca_pred = lda_pca.predict(X_pca_test)

# Evaluate the accuracy of the classifier
accuracy_LDA2_pca = accuracy_score(Y_pca_test, Y_pca_pred)
print(f"Accuracy: {accuracy_LDA2_pca:.2f}")
```

The code performs principal component analysis (PCA) on the original dataset _X_, with the aim of reducing the dimensionality of the dataset from the original number of features to just 10. The PCA() function is used to create a PCA object, and _n_components=10_ is set to specify that we want to keep only the top 10 principal components. The .fit() method is then called on the PCA object with _X_ as input to train the PCA model, and the .transform() method is called to transform _X_ into a new dataset _X_pca_ with **only 10 dimensions**. 

We filter the dataset _X_pca_ to only include samples that correspond to the digit 4 or the digit 9, and the corresponding labels _Y_pca_ are also filtered accordingly. The filtered dataset and labels are then split into training and testing sets using the _train_test_split()_ function from the _sklearn.model_selection_ module. The test_size parameter is set to 0.2 to specify that we want to use 20% of the data for testing, and random_state is set to 42 to ensure reproducibility.

A linear discriminant analysis (LDA) model is then trained on the training set using the transformed dataset _X_pca_train_ and corresponding labels _Y_pca_train_. The LinearDiscriminantAnalysis() function from the sklearn.discriminant_analysis module is used to create the LDA object, and the .fit() method is called to train the LDA model. The trained LDA model is used to make predictions on the testing set _X_pca_test_, and the predicted labels are stored in _Y_pca_pred_.

Finally, the accuracy of the classifier is evaluated by comparing the predicted labels _Y_pca_pred_ with the true labels _Y_pca_test_, and computing the accuracy using the _accuracy_score()_ function from the _sklearn.metrics_ module. The resulting accuracy is printed to the console.

The same was then done for the comparison and classification of three digits, 0, 8, and 9. The approach is entirely similar except now we must filter out the 0s, 8s, and 9s into the dataset under study:

```
# LDA for 3 digits
# Select only 0s and 8s and 9s
X_089 = X_pca[(Y == 0) | (Y == 8) | (Y == 9)]
Y_089 = Y[(Y == 0) | (Y == 8) | (Y == 9)]
```

After that, I looped through all possible combinations of digits and calculated the accuracy for all 45 unique combinations:

```
from itertools import combinations

# Define the list of digits to use
digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Initialize a list to store the accuracies and corresponding digit pairs
results = []

# Loop over all pairs of digits
for digit1, digit2 in combinations(digits, 2):
    
    # Select the data for the current pair of digits
    X_pair = X_pca[(Y == digit1) | (Y == digit2)]
    Y_pair = Y[(Y == digit1) | (Y == digit2)]
    
    # Split into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_pair, Y_pair, test_size=0.2, random_state=42)
    
    # Train an LDA model
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, Y_train)
    
    # Make predictions on the testing set
    Y_pred = lda.predict(X_test)
    
    # Evaluate the accuracy of the classifier
    accuracy = accuracy_score(Y_test, Y_pred)
    
    # Append the digit pair and accuracy to the results list
    results.append((digit1, digit2, accuracy))
```

The approach is the same as the two digits version execept we loop through different combinations of digits and store the pair and their accuracy in a _results_ array. I then sorted this _results_ array by accuracy and printed the sorted array:

```
# sort the results
sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
# Print the sorted list of results
for result in sorted_results:
    print(f"Digits: ({result[0]}, {result[1]})\tAccuracy: {result[2]:.2f}")
```

The dataset was then tested for accuracy when using SVM and decision tree classifiers. The approach is the same as before except rather than fitting our data to LDA, we fit to SVC (SVM) and DecisionTreeClassifier (DTC):

```
#Split the data into training and testing sets
X_train_SVM, X_test_SVM, y_train_SVM, y_test_SVM = train_test_split(X_pca, Y, test_size=0.2, random_state=42)

#Scale the data using a standard scaler
scaler = StandardScaler() 
X_train_SVM = scaler.fit_transform(X_train_SVM) 
X_test_SVM = scaler.transform(X_test_SVM)

#Initialize an SVM classifier
svm = SVC(kernel='rbf', C=1, gamma='auto')

# Train the classifier on the training set
svm.fit(X_train_SVM, y_train_SVM)

# Make predictions on the testing set
y_pred_SVM = svm.predict(X_test_SVM)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test_SVM, y_pred_SVM) 
print(f"Accuracy: {accuracy:.2f}")
```

```
# Split the data into training and testing sets
X_train_DTC, X_test_DTC, y_train_DTC, y_test_DTC = train_test_split(X_pca, Y, test_size=0.2, random_state=42)

# Train a decision tree classifier with default hyperparameters
clf = DecisionTreeClassifier()
clf.fit(X_train_DTC, y_train_DTC)

# Make predictions on the testing set
y_pred_DTC = clf.predict(X_test_DTC)

# Evaluate the accuracy of the classifier
accuracy_DTC = accuracy_score(y_test_DTC, y_pred_DTC)
print(f"Accuracy: {accuracy:.2f}")
```

I then printed this decision tree using the _plot_tree_ function:

```
# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True)
plt.show()
```

The _filled_ parameter was set to true since that fils the decision tree nodes with color for better visuilization of the classification.

Finally, the accuracy of classification of the least and most common pairs of digits (according to LDA) was compared to the accuracy when using SVM and CLF:

```
# Hardest pair: (4,9)
# Easiest pair: (1,0)
# Select only 4s and 9s
X_49 = X_pca[(Y == 4) | (Y == 9)]
Y_49 = Y[(Y == 4) | (Y == 9)]

# Split the data into training and testing sets
X_train_SVM_49, X_test_SVM_49, y_train_SVM_49, y_test_SVM_49 = train_test_split(X_49, Y_49, test_size=0.2, random_state=42)

# Scale the data using a standard scaler
scaler_49 = StandardScaler()
X_train_SVM_49 = scaler_49.fit_transform(X_train_SVM_49)
X_test_SVM_49 = scaler_49.transform(X_test_SVM_49)

# Initialize an SVM classifier
svm_49 = SVC(kernel='rbf', C=1, gamma='auto')

# Train the classifier on the training set
svm_49.fit(X_train_SVM_49, y_train_SVM_49)

# Make predictions on the testing set
y_pred_SVM_49 = svm_49.predict(X_test_SVM_49)

# Evaluate the accuracy of the classifier
accuracy_SVM_49 = accuracy_score(y_test_SVM_49, y_pred_SVM_49)
print(f"Accuracy: {accuracy_SVM_49:.2f}")
```

```
# Split the data into training and testing sets
X_train_DTC_49, X_test_DTC_49, y_train_DTC_49, y_test_DTC_49 = train_test_split(X_49, Y_49, test_size=0.2, random_state=42)

# Train a decision tree classifier with default hyperparameters
clf_49 = DecisionTreeClassifier()
clf_49.fit(X_train_DTC_49, y_train_DTC_49)

# Make predictions on the testing set
y_pred_DTC_49 = clf.predict(X_test_DTC_49)

# Evaluate the accuracy of the classifier
accuracy_DTC_49 = accuracy_score(y_test_DTC_49, y_pred_DTC_49)
print(f"Accuracy: {accuracy_DTC_49:.2f}")
```

```
X_10 = X_pca[(Y == 1) | (Y == 0)]
Y_10 = Y[(Y == 1) | (Y == 0)]

# Split the data into training and testing sets
X_train_SVM_10, X_test_SVM_10, y_train_SVM_10, y_test_SVM_10 = train_test_split(X_10, Y_10, test_size=0.2, random_state=42)

# Scale the data using a standard scaler
scaler_10 = StandardScaler()
X_train_SVM_10 = scaler_10.fit_transform(X_train_SVM_10)
X_test_SVM_10 = scaler_10.transform(X_test_SVM_10)

# Initialize an SVM classifier
svm_10 = SVC(kernel='rbf', C=1, gamma='auto')

# Train the classifier on the training set
svm_10.fit(X_train_SVM_10, y_train_SVM_10)

# Make predictions on the testing set
y_pred_SVM_10 = svm_10.predict(X_test_SVM_10)

# Evaluate the accuracy of the classifier
accuracy_SVM_10 = accuracy_score(y_test_SVM_10, y_pred_SVM_10)
print(f"Accuracy: {accuracy_SVM_10:.2f}")
```

```
# Split the data into training and testing sets
X_train_DTC_10, X_test_DTC_10, y_train_DTC_10, y_test_DTC_10 = train_test_split(X_10, Y_10, test_size=0.2, random_state=42)

# Train a decision tree classifier with default hyperparameters
clf_10 = DecisionTreeClassifier()
clf_10.fit(X_train_DTC_10, y_train_DTC_10)

# Make predictions on the testing set
y_pred_DTC_10 = clf.predict(X_test_DTC_10)

# Evaluate the accuracy of the classifier
accuracy_DTC_10 = accuracy_score(y_test_DTC_10, y_pred_DTC_10)
print(f"Accuracy: {accuracy_DTC_10:.2f}")
```

The only difference in these cases versus the previously described cases is that we must filter different values from our main PCA space dataset.

The results will be analyzed in depth in Section IV.

### Sec. IV. Computational Results

After running SVM, the singular values and the cumulative sum of the singular values were plotted against the singular value index:

![image](https://user-images.githubusercontent.com/116219100/234247692-6b79498d-9172-4157-b17c-c23ac8e56fac.png)
*Figure 2: Singular Value vs Singular Value Index*

![image](https://user-images.githubusercontent.com/116219100/234247748-91ff782d-5d6a-4227-b697-a6eb1fed7c08.png)
*Figure 3: Cumulative Sum of Singular Value vs Singular Value Index*
 
Figures 2 and 3 show us exactly what we would expect from singular values. We observe that they decrease in value as we move down the diagonal of the matrix _s_. We also see the how much higher the first couple singular values are compared to the rest. This proves that the first couple singular values are all we need to properly define this data set since the effect of the latter singular values present little to no variance and thus do not alter our model much. The cumulative sum plot in figure 3 helps visualize this effect more strongly.

![image](https://user-images.githubusercontent.com/116219100/234249335-d69da457-8156-4fd3-862a-e87f2d243bd4.png)
*Figure 4: Scree Plot of Percentage of Total Variance Captured vs Principal Component*

To stress on this concept more, I have made a scree plot of the cumulative sum plot in figure 3. Here we can see that 'elbow' more clearly which is the region where the percentage of total variance captured begins to add little to no amount. Visually, it seemed to me that this elbow occurs at principal component 80 and thus I drew a vertical line at that point. Notice how basically after 80 principal components the variation of results is very slim and thus those values can be truncated. This means our data can be efficiently represented in 80 dimensions of V-modes rather than the 784 pixel space or 784 V-modes. Thus, we could have used the _TruncatedSVD()_ function with _n_components = 80_ instead of the regular SVD. The truncated SVD approach would have saved ample time since our feature space would have been reduced by 10-fold. This would result in much quicker runtime at the cost of accuracy. However, the difference in accuracy would be very minimal as the plot in figure 4 tells us since 80 PC components is somewhat of a cuttoff point.



### Sec. V. Summary and Conclusions
