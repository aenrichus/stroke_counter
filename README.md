# stroke_counter
This model consists of a convolutional neural network used to count the number of strokes Japanese Kanji.

## input
Images of individual kanji characters of shape 64x64. These are divided to the characters used in daily life that must be learned in schools (joyo) and those used primarily in names (jinmeiyo).

## output
Number of strokes in each characters as one-hot nodes.

## performance
The model is capable of learning all of the items in the training set, but has difficulty generalizing to more complex characters.
