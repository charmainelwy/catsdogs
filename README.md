# Cats and dogs classification using CNNs 

Welcome! This is my first deep learning project and it is going to be a simple kernel on image recognition. Feel free to comment and make any changes, I am open to any feedback :) 

## Problem statement: 
I will be implementing Convolution Neural Network (CNN) Classifier to predict the category of dog or cat. I used the Asirra dataset, which can be found here. https://www.kaggle.com/c/dogs-vs-cats
I will be using Python on VSC. 

Let's look at our data. 
```
np.shape(df)
```
"*(25001, 2)*"
There are 25001 rows and 2 columns in our train dataset. Data seems well balanced. 

![Screenshot](Screenshot1.png)



## Model building 

#### 1. Convolutional layer 
I believe that convolutional layers are filters, which are the core building blocks of convolutional neural networks. Different filters pick out different parts of an image (example vertical lines or horizontal lines). More layers will help the model learn more complicated features. 
I used conv2D() function from Keras to build my first convolutional layer. The number of feature detectors/filters is set to be 32, and each filter dimension is (3,3). 
input_shape is the shape of input images on which we apply feature detectors through convolution. It is set to (64, 64, 3). 3 is the number of channels for a colored image, (64, 64) is the image dimension for each channel. 
The last argument is the activation function, which is set to 'ReLU' to remove any negative pixel values in feature map. This will add non-linearity to a non-linear classification problem. 

#### 2. Batch normalization 
Batch normalization layer is a method to make the network faster and more stable through normalization of the input layer by recentering and rescaling. This has the effect of stabilizing the learning process and dramatically reducing the number of training epochs required to train deep networks. 

#### 3. Pooling layers 
Pooling layers reduce the input sizes of an image by downsampling to reduce the number of parameters and computation in the network. This is done without losing key features and spatial structure information in the images. 
A pooling layer follows after each convolutional layer, which performs a MAX operation which means it selects the maximum value inside each 2 x 2 matrix since I chose pool_size=(2,2).

#### 4. Dropout layer 
A dropout layer disregards some of the neurons as a form of regularization to prevent overfitting. 

#### 5. Flattening 
Finally, a flatten layer helps to pass output into a regular MLP. 

#### 6. Full connected layer 
With the above layers, we converted an image into a one-dimensional vector. Now, we will create a hidden layer. "*output_dim*" is the number of nodes in hidden layer. I used 512 and "*ReLU*" as activation function. 

#### 7. Output layer 
Then, I added an output layer. For binary classification, "*output_dim*" is 1, and activation functioon is "*Sigmoid*". 

```
model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(64,64,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))
```


## Model compiling 
With all layers added, i compiled the CNN with the parameters optimizer='adam', loss='binary_crosssentropy' for binary classification and metrics='accuracy'. 

```
model.compile(loss='binary_crossentropy',optimizer='adam',metrics='accuracy')
```


## Model fitting 
I am aware that we use image augmentation (rotating, flipping etc) to increase the number of images for the training set to prevent overfitting. This splits images into different batches, and each batch will be applied random image transformation on a random selection of images to create many more images. 
I use flow_from_directory method to load images and apply image augmentation. The target size is set to be (64,64), same shape as convolutional layer. 

```
train_datagen = ImageDataGenerator(rotation_range=15,rescale=1./255,shear_range=0.1,zoom_range=0.2,
                                horizontal_flip=True,width_shift_range=0.1,height_shift_range=0.1 )

train_generator = train_datagen.flow_from_dataframe(train_df, "/Users/admin/Documents/cat dog/train", 
    x_col='filename',y_col='category',target_size=(64,64),
    class_mode='binary',batch_size=batch_size)

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(validate_df, 
    "/Users/admin/Documents/cat dog/train", x_col='filename',y_col='category',
    target_size=(64,64),class_mode='binary',batch_size=batch_size
)

test_datagen = ImageDataGenerator(rotation_range=15,rescale=1./255,shear_range=0.1,zoom_range=0.2,
                                horizontal_flip=True,width_shift_range=0.1,height_shift_range=0.1)

test_generator = train_datagen.flow_from_dataframe(train_df,"/Users/admin/Documents/cat dog/test1",
    x_col='filename',y_col='category',target_size=(64,64),
    class_mode='binary',batch_size=batch_size)
```

Now, let's fit and test the model. I obtained a training accuracy of __ and test accuracy of __ . The scores are reasonably close, suggesting the model is probably not over or underfit.


## Model improvements 
We should always strive to improve accuracy and reduce overfitting. This is by either adding more convolution layers or adding more dense layers.

