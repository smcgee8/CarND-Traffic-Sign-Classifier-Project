# **Traffic Sign Recognition**

## Writeup

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the NumPy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is (32,32)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

For the visualization, I simply displayed a randomly selected image from the dataset. By running this code-block multiple times, I was able to see various images and get a sense for the type of data my model would be training on. Here's one example:

![Example Image from Training Set][sample_image.png]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I normalized the images by subtracting the mean and dividing by the standard deviation. Normalization is important because images were taken under a variety of different lighting conditions (and possibly also with different cameras), so it is helpful to recast the data to be between -1 and 1 for all images. I initially wanted to subtract the actual mean for each layer and divide by the actual standard deviation by iterating over the images. However, while I did use the actual mean, I simply divided by 128 as a proxy for the standard deviation. When I tried to divide by the actual standard deviation, I got value warnings (I'm guessing because of divide-by-zero issues). I'm still not entirely sure how to fix that, but dividing by 128 seemed to improve the model performance.

I ended up not augmenting the training set, but I attempted to do so to see what would happen. My hypothesis was that I could improve model performance by taking the entire training set, rotating it by 180 degrees, and appending it to the end of the training set (effectively doubling the size of the training data). I thought more variation in my training set would help improve my model. However, there did not appear to be any difference in model convergence, and the processing took considerably more time with the additional images. I still think a different type of augmentation could be helpful, but I'm not sure why this particular attempt was unsuccessful.

As a last step, I shuffled the training data to ensure that model training performance was not dependent on the ordering of the data.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x50 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x50 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x100 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x100 				|
| Flatten		| outputs 2,500       									|
| Fully connected		| outputs 400        									|
| RELU					|												|
| Dropout					|	Keep prob: 50%											|
| Fully connected		| outputs 120        									|
| RELU					|												|
| Dropout					|	Keep prob: 50%											|
| Fully connected		| outputs 84        									|
| RELU					|												|
| Fully connected		| outputs 43        									|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the AdamOptimizer (same as for LeNet). I used 15 EPOCHS, a batch size of 128, and a learning rate of 0.001. As mentioned in my Model stack, above, I was able to improve model performance by adding a 50% drop-out to the first two fully-connected layers.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy reached 0.977 during the final EPOCH
* test set accuracy (tested once after all model tweaks were completed) was 0.954

It should be noted that dropout was applied during the training process but the "keep probability" was changed to 1.0 during validation and test evaluations.

In creating my model, I took the following steps:
* I started with the LeNet architecture from the previous lab and modified it as little as possible (e.g. adding 3 channel support) to make it run for the traffic data. That got me to ~0.88 on the validation data. I think LeNet was a good starting choice because it's a proven architecture for recognizing simple shapes in images. Traffic signs are certainly more complicated than recognizing digits (for example, there are now 43 classes instead of 10), however, the problem is at least on a similar scale.
* I then added normalization, first attempting a simple "subtract 128 and divide by 128" heuristic, and then moving to subtracting the mean of each image channel because it offered slightly better performance. Normalization got me to ~0.91 on the validation data. I decided not to grayscale the images because it seemed to me that the color information was actually a valuable way to distinguish signs (whereas for digits, for example, color is less relevant).
* I modified the structure of the LeNet network slightly, increasing the number of filters for the Conv layers and adding an additional fully-connected layer. This seemed to improve performance slightly (to ~0.93). My assumption is that the previous model was underfitting the data and that this is why I was able to improve the performance by increasing the complexity of the model.
* My next step was to try training set augmentation by copying the entire training set, rotated by 180 degrees. This did not seem to improve model performance (in fact it seemed to slightly decrease performance) and slowed the training process considerably, so I ended up reversing this decision (the code is still there in the preprocessing section, it's just commented out).
* My last adjustment was adding drop-out for some of the fully-connected layers. The negative impact of adding drop-out was that the model took longer to converge and would actually decrease in performance between iterations at times, but the model ultimate converged to better performance. To account for the slower convergence progress, I increased the number of EPOCHS to 15 from the original LeNet model (which used 10).

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Image 1][web_german_traffic_signs/1.jpeg] ![Image 2][web_german_traffic_signs/2.jpg] ![Image 3][web_german_traffic_signs/3.jpeg] ![Image 4][web_german_traffic_signs/4.jpg] ![Image 5][web_german_traffic_signs/5.jpeg]

* Image 1: One challenge with this image is that there are lots of german traffic signs with red triangles that refer to different types of warnings. Would the model realize that a red triangle with nothing in the middle and the point facing downward corresponds to "yield"?
* Image 2: The challenge with this image is that there are many other traffic signs involving a blue circle with different types of arrows. Would the model be able to distinguish this one from the others?
* Image 3: Similar to image 2.
* Image 4: This image was sufficiently distinct. I thought it could be the easiest of the set to classify.
* Image 5: Similar to image 1. There are lots of signs with red triangles oriented in this fashion with some sort of illustration in the middle. Would it be able to distinguish that this is road work?

A final, overarching, concern I had for all these images is that if I chose a sign that wasn't actually a german traffic sign (it can be difficult to tell on Google Images) or happened to not be included in the training set, the model would not be able to classify it.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Yield      		| Yield   									|
| Roundabout mandatory     			| Roundabout mandatory 										|
| Ahead only					| Ahead only											|
| No entry	      		| No entry					 				|
| Road work			| Road work      							|


The model was able to correctly guess all of the traffic signs, which gives an accuracy of 100%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

Image 1

| Logits         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 65         			| Yield   									|
| 11     				| Priority road 										|
| 9					| 36											|
| 8	      			| 15					 				|
| 8				    | 10      							|

Image 2

| Logits         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 20         			| 40   									|
| 8     				| 35 										|
| 5					| 37											|
| 4	      			| 34					 				|
| 4				    | 38      							|

Image 3

| Logits         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 27         			| 35   									|
| 10     				| 34 										|
| 7					| 36											|
| 5	      			| 3					 				|
| 5				    | 33      							|

Image 4

| Logits         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 69         			| 35   									|
| 36     				| 34 										|
| 21					| 36											|
| 16	      			| 3					 				|
| 12				    | 33      							|

Image 3

| Logits         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 27         			| 35   									|
| 10     				| 34 										|
| 7					| 36											|
| 5	      			| 3					 				|
| 5				    | 33      							|
