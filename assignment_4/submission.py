from numpy.core.fromnumeric import argmax
from numpy.lib.function_base import vectorize
import math
import time
from collections import Counter
import numpy as np
from numpy.core.arrayprint import dtype_is_implied
np.nanmedian


class DecisionNode:
    """Class to represent a single node in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """Create a decision function to select between left and right nodes.
        Note: In this representation 'True' values for a decision take us to
        the left. This is arbitrary but is important for this assignment.
        Args:
            left (DecisionNode): left child node.
            right (DecisionNode): right child node.
            decision_function (func): function to decide left or right node.
            class_label (int): label for leaf node. Default is None.
        """

        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Get a child node based on the decision function.
        Args:
            feature (list(int)): vector for feature.
        Return:
            Class label if a leaf node, otherwise a child node.
        """

        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.
    Args:
        data_file_path (str): path to data file.
        class_index (int): slice output by index.
    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if(class_index == -1):
        classes = out[:, class_index]
        features = out[:, :class_index]
        return features, classes
    elif(class_index == 0):
        classes = out[:, class_index]
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the sample data.
    Tree is built fully starting from the root.
    Returns:
        The root node of the decision tree.
    """
    decision_tree_root = None
    a1 = DecisionNode(None, None, lambda feature: feature[0] == 0, None)
    a2 = DecisionNode(None, None, lambda feature: feature[1] == 0, None)
    a3 = DecisionNode(None, None, lambda feature: feature[2] == 1, None)
    a4 = DecisionNode(None, None, lambda feature: feature[3] == 0, None)

    a1.right = DecisionNode(None, None, None, 1)
    a1.left = a2

    a2.right = a3
    a2.left = a4

    a3.right = DecisionNode(None, None, None, 1)
    a3.left = DecisionNode(None, None, None, 0)

    a4.right = DecisionNode(None, None, None, 1)
    a4.left = DecisionNode(None, None, None, 0)

    decision_tree_root = a1

    return decision_tree_root


def confusion_matrix(classifier_output, true_labels):
    """Create a confusion matrix to measure classifier performance.
    Output will in the format:
        [[true_positive, false_negative],
         [false_positive, true_negative]]
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        A two dimensional array representing the confusion matrix.
    """
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for i in range(len(classifier_output)):
        if classifier_output[i] and true_labels[i]:
            true_positive += 1
        elif not classifier_output[i] and not true_labels[i]:
            true_negative += 1
        elif (classifier_output[i] and not true_labels[i]):
            false_positive += 1
        else:
            false_negative += 1

    return [[true_positive, false_negative], [false_positive, true_negative]]


def precision(classifier_output, true_labels):
    """Get the precision of a classifier compared to the correct values.
    Precision is measured as:
        true_positive/ (true_positive + false_positive)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The precision of the classifier output.
    """
    cf_matrix = confusion_matrix(classifier_output, true_labels)
    true_positive = cf_matrix[0][0]
    false_positive = cf_matrix[1][0]
    precision = true_positive / (true_positive + false_positive)

    return precision


def recall(classifier_output, true_labels):
    """Get the recall of a classifier compared to the correct values.
    Recall is measured as:
        true_positive/ (true_positive + false_negative)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The recall of the classifier output.
    """
    cf_matrix = confusion_matrix(classifier_output, true_labels)
    true_positive = cf_matrix[0][0]
    false_negative = cf_matrix[0][1]
    recall = true_positive / (true_positive + false_negative)

    return recall


def accuracy(classifier_output, true_labels):
    """Get the accuracy of a classifier compared to the correct values.
    Accuracy is measured as:
        correct_classifications / total_number_examples
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The accuracy of the classifier output.
    """
    cf_matrix = confusion_matrix(classifier_output, true_labels)
    true_positive = cf_matrix[0][0]
    true_negative = cf_matrix[1][1]
    false_positive = cf_matrix[1][0]
    false_negative = cf_matrix[0][1]
    total = true_positive + true_negative + false_negative + false_positive
    accuracy = (true_negative + true_positive) / total

    return accuracy


def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.
    Args:
        class_vector (list(int)): Vector of classes given as 0 or 1.
    Returns:
        Floating point number representing the gini impurity.
    """
    array = np.array(class_vector)
    bin_count = np.bincount(array)
    p_0 = bin_count[0] / (np.sum(bin_count))
    p_1 = 1 - p_0

    return (1 - p_0 * p_0 - p_1 * p_1)


def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0 or 1.
        current_classes (list(list(int): A list of lists where each list has
            0 and 1 values).
    Returns:
        Floating point number representing the information gain.
    """
    giniA = gini_impurity(previous_classes)
    sA = len(previous_classes)

    current_classes = np.array(current_classes)
    shape = np.shape(current_classes)
    rows = shape[0]
    gini_array = []
    for i in range(0, rows):
        array = current_classes[i]
        giniS = gini_impurity(array)
        sV = len(array)
        ratio = sV / sA
        gini_array.append(ratio * giniS)

    gini_array = np.array(gini_array)
    gini_gain = giniA - np.sum(gini_array)

    return gini_gain


def getBinary(features):
    """Convert the features vector from continous to discrete values
    Args:
        features: numpy vector of floating point values
    Returns:
        binary_features: numpy vector of discrete values found using
        Gaussian distribution
    """
    f_mean = np.mean(features)
    f_std = np.std(features)
    mean_plus_std = f_mean + f_std
    mean_mins_std = f_mean - f_std

    # generate a vector of the same size as features
    binary_array = np.zeros(len(features), int)

    for i in range(0, len(features)):
        # classify as 0 for within 1 std and 1 for everything else
        value = features[i]
        if value > mean_plus_std or value < mean_mins_std:
            binary_array[i] = 1

    return binary_array


def buildTree(features, classes, depth, max_depth):
    """Build tree that automatically finds the decision functions.
    Args:
        features (m x n): m examples with n features.
        classes (m x 1): Array of Classes.
        depth (int): depth to build tree to.
    Returns:
        Root node of decision tree.
    """

    features = np.array(features, int)
    classes = np.array(classes, int)
    print(classes)

    # when no feature is left to classify
    if np.size(features) < 1:
        bin_count = np.bincount(classes)
        most_freq_class = np.argmax(bin_count)
        return DecisionNode(None, None, None, most_freq_class)

    # when leaf node
    unique_elements = np.unique(classes)
    if len(unique_elements) == 1:
        return DecisionNode(None, None, None, classes[0])

    # max depth reached
    if depth == max_depth:
        bin_count = np.bincount(classes)
        most_freq_class = np.argmax(bin_count)
        return DecisionNode(None, None, None, most_freq_class)

    gini_list = []
    for n in range(0, np.shape(features)[1]):
        previous_classes = classes        # get the first column
        feature_T = []
        feature_F = []
        for i in range(len(previous_classes)):
            if(features[i][n] == previous_classes[i]):
                feature_T.append(features[i][n])
            else:
                feature_F.append(features[i][n])
        current_classes = [feature_T, feature_F]
        feature_gain = gini_gain(previous_classes, current_classes)
        gini_list.append(feature_gain)
    gini_list = np.array(gini_list)
    idx = argmax(gini_list)

    # now split along the feature with this index
    features_0 = []
    features_1 = []
    classes_0 = []
    classes_1 = []
    for i in range(0, len(classes)):
        # for 0
        if(features[i][idx] == 0):
            features_0.append(features[i])
            classes_0.append(classes[i])
        else:
        # for 1
            features_1.append(features[i, ])
            classes_1.append(classes[i])

    features_0 = np.array(features_0, int)
    features_1 = np.array(features_1, int)
    classes_0 = np.array(classes_0, int)
    classes_1 = np.array(classes_1, int)

    try:
        if np.size(features_0) >= 1:
            features_0 = np.delete(features_0, idx, 1)
        if np.size(features_1) >= 1:
            features_1 = np.delete(features_1, idx, 1)
    except:
        print("wtf")


    if np.size(classes_0) > 0 and np.size(classes_1) > 0:
        return DecisionNode(buildTree(features_0, classes_0, depth + 1, max_depth), 
                            buildTree(features_1, classes_1, depth + 1, max_depth),
                            lambda x:x[idx] == 0, None)
    elif np.size(classes_0) < 1:
        return DecisionNode(None,
                            buildTree(features_1, classes_1, depth + 1, max_depth),
                            lambda x:x[idx] == 0, None)
    else:
        return DecisionNode(buildTree(features_0, classes_0, depth + 1, max_depth),
                            None,
                            lambda x:x[0] == 0, None)


class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=float('inf')):
        """Create a decision tree with a set depth limit.
        Starts with an empty root.
        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
            depth (int): depth to build tree to.
        Returns:
            Root node of decision tree.
        """
        # converting from continous to discrete
        features = np.array(features)
        classes = np.array(classes, int)
        features_t = np.transpose(features)
        binary_features = np.zeros(np.shape(features_t), int)
        for i in range(0, np.shape(features_t)[0]):
            binary_features[i] = getBinary(features_t[i])
        binary_features = np.transpose(binary_features)

        # Now build tree
        root = buildTree(binary_features, classes, depth, self.depth_limit)
        self.root = root
        return root

    def classify(self, features):
        """Use the fitted tree to classify a list of example features.
        Args:
            features (m x n): m examples with n features.
        Return:
            A list of class labels.
        """

        class_labels = []
        features = features.shape

        # TODO: finish this.
        raise NotImplemented()
        return class_labels


def generate_k_folds(dataset, k):
    """Split dataset into folds.
    Randomly split data into k equal subsets.
    Fold is a tuple (training_set, test_set).
    Set is a tuple (features, classes).
    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.
    Returns:
        List of folds.
        => Each fold is a tuple of sets.
        => Each Set is a tuple of numpy arrays.
    """

    # TODO: finish this.
    raise NotImplemented()


class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees, depth_limit, example_subsample_rate,
                 attr_subsample_rate):
        """Create a random forest.
         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """

        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        # TODO: finish this.
        raise NotImplemented()

    def classify(self, features):
        """Classify a list of features based on the trained random forest.
        Args:
            features (m x n): m examples with n features.
        """

        # TODO: finish this.
        raise NotImplemented()


class ChallengeClassifier:
    """Challenge Classifier used on Challenge Training Data."""

    def __init__(self):
        """Create challenge classifier.
        Initialize whatever parameters you may need here.
        This method will be called without parameters, therefore provide
        defaults.
        """

        # TODO: finish this.
        raise NotImplemented()

    def fit(self, features, classes):
        """Build the underlying tree(s).
            Fit your model to the provided features.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        # TODO: finish this.
        raise NotImplemented()

    def classify(self, features):
        """Classify a list of features.
        Classify each feature in features as either 0 or 1.
        Args:
            features (m x n): m examples with n features.
        Returns:
            A list of class labels.
        """

        # TODO: finish this.
        raise NotImplemented()


class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be added to array.
        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Element wise array arithmetic using vectorization.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be sliced and summed.
        Returns:
            Numpy array of data.
        """
        np_data = np.array(data)
        return (np.add(np.multiply(data, data), data))

    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be added to array.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """
        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return max_sum, max_sum_index

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be sliced and summed.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """
        data = np.array(data)
        sum_array = np.zeros(100)
        for i in range(0, 100):
            sum_array[i] = sum(data[i])

        return (np.amax(sum_array), np.argmax(sum_array))

    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        unique_dict = {}
        flattened = np.hstack(data)
        for item in range(len(flattened)):
            if flattened[item] > 0:
                if flattened[item] in unique_dict:
                    unique_dict[flattened[item]] += 1
                else:
                    unique_dict[flattened[item]] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """
        np_data = (np.array(data, int))
        flat_data = np_data.flatten()
        flat_data = np.array([value for value in flat_data if value > 0])
        bin_data = np.bincount(flat_data)
        unique_dict = {}

        counter = 0
        for value in bin_data:
            if(value != 0):
                unique_dict[counter] = bin_data[counter]
            counter += 1

        return list(unique_dict.items())


def return_your_name():
    # return your name
    return "Manan Patel"


# tests
# load the data
data = load_csv("part23_data.csv")
features = data[0]
classes = data[1]
classes = np.array(classes)
print(np.shape(classes))
tree = DecisionTree()
root = tree.__build_tree__(features, classes, 0)
print(3)