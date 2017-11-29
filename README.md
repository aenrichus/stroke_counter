# Kanji Stroke Counter
These models are use to determine the number of strokes in Japanese kanji. The primary model consists of a convolutional neural network, but a second model uses only a feedforward network.

## Version History
v0.2 - This version included a number of key changes including the addition of a baseline feedforward network. It also includes significant changes to the model input and output. The input now uses all characters and splits them 80/20 into training and test sets. The output now include much more detailed information about the character, number of strokes, and prediction. An R script is included for cleaning and summarizing the data.

v0.1 - The basic convolutional model was completed about two years ago and was written for version 0.x of TensorFlow. It was intended as a proof of concept and was moderately successful.

## Upcoming Tasks
- Set a constant random seed for better comparison
- Generate images in different fonts during the training process.
- Perform a selectivity analysis to determine whether the networks are learning sub-lexical radicals.
- Create data sets for simplified and traditional Chinese characters.
- Add the ability to grow vocabulary (curriculum learning).

## Input
Images of individual kanji characters of shape 32x32. Included are the characters used in daily life that must be learned in schools (joyo) and those used primarily in names (jinmeiyo). Combined, they make up a somewhat limited training set for AI applications, but are rather similar to the characters readily known by many Japanese adults.

## Output
Predicted number of strokes in each characters as one-hot nodes.

## Performance
After 5 million trials, the baseline feedforward model reaches 99.8% accuracy on the training set but only 22% accuracy on the test set. The convolutional neural network learns more quickly, but shows similar final accuracy on the training (99.96%) and test (28%) sets. An examination of the errors may shine light on where and why they are being made.

Poor generalization may be due to having a limited number of examples for each number of strokes. Possible solutions to this issue might include:
- Train on a larger set of kanji characters, e.g. all included in Kanji Kentei level 1.
- Use many more fonts to add variability and increase the number of training examples.
- Train on fake characters composed of radicals not usually combined to increase the number of training examples.
- Use different hyperparameter settings, e.g. decreasing the number of units in the FC layer.
- Use additional targets, e.g. including radicals as part of the target that must be learned.
- Use different network architecture, e.g. CapsNet.
