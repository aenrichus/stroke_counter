# Kanji Stroke Counter
These models are use to determine the number of strokes in Japanese kanji. The primary model consists of a convolutional neural network, but a second model uses only a feedforward network.

## Version History
v0.2 - This version included a number of key changes including the addition of a baseline feedforward network. It also includes significant changes to the model input and output. The input now uses all characters and splits them 80/20 into training and test sets. The output now include much more detailed information about the character, number of strokes, and prediction. An R script is included for cleaning and summarizing the data.
v0.1 - The basic convolutional model was completed about two years ago and was written for version 0.x of TensorFlow. It was intended as a proof of concept and was moderately successful.

## Upcoming Tasks
- Generate images in different fonts during the training process.
- Perform a selectivity analysis to determine whether the networks are learning sub-lexical radicals.
- Create data sets for simplified and traditional Chinese characters.

## input
Images of individual kanji characters of shape 32x32. Included are the characters used in daily life that must be learned in schools (joyo) and those used primarily in names (jinmeiyo). Combined, they make up a somewhat limited training set for AI applications, but are rather similar to the characters readily known by many Japanese adults.

## output
Predicted number of strokes in each characters as one-hot nodes.

## performance
This section will be updated after further analysis. Preliminary results are promising.
