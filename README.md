# Microexpression Detection
## What is a microexpression
A microexpression is a facial expression that only lasts for a short moment.
Microexpressions, arguably, express 7 emotions. In my case, I trained the model
to detect emotions such as: anger, disgust, fear, happiness, neutral, sadness, and surprise.
For the dataset and further read, you may want to visit: 
https://www.kaggle.com/datasets/kmirfan/micro-expressions?resource=download

## How to use

### Creating the model

First, we collect all the images, get the grayscale of both original and rotated version of them
with the function _create_the_model_. After that, we store features and labels , then turn them into a
numpy array. Finally, we save the model with _pickle_.

### Training the model

After loading the model, we create our convolutional layers, pooling layers (max pooling), and
fully connected layers. For the convolutional layer 32 filters and (3,3) kernel size were chosen.
The last FC layer has also a dropout. Then we compile the model with _adam_ optimizer and
_sparse_categorical_crossentropy_. Finally, save the model with _Sequential.save()_
 
### Prediction
In the final part, we define a function for resizing and turning to grayscale the given image.
After having the prepared image, we use _model.predict_ to make a prediction with the model which
we trained in the previous part.




