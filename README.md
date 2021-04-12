# Cats and dogs classification using CNNs 

Welcome! This is my first deep learning project and it is going to be a simple kernel on image recognition. Feel free to comment and make any changes, I am open to any feedback :) 

## Problem statement: 
I will be implementing Convolution Neural Network (CNN) Classifier to predict the category of dog or cat. I used the Asirra dataset, which can be found here. https://www.kaggle.com/c/dogs-vs-cats
I will be using Python on VSC. 

###### Creating training data
```
filenames=os.listdir("/Users/admin/Documents/cat dog/dogsvscats/train")

categories=[]
for filename in filenames:
    category=filename.split('.')[0]
    if category=='dog':
        categories.append("dog")
    else:
        categories.append("cat")

df=pd.DataFrame({
    'filename':filenames,
    'category':categories })

np.shape(df)
```
> (25001, 2)

There are 25001 images in our train dataset. 12501 cats and 12500 dogs. Data seems well balanced to proceed. 
<img width="393" alt="Screenshot1" src="https://user-images.githubusercontent.com/61202712/114333550-2640dd80-9b7b-11eb-9abc-79d97afc1365.png">

Now, we will create a new dataset containing 2 subsets, a training set with 10,000 samples of each class (20,000 in total) and a validation dataset with 5001 total.
```
train_df, validate_df = train_test_split(df, test_size = 0.2, random_state = 42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
total_train=train_df.shape[0]  # 20 000 
total_validate=validate_df.shape[0]  #5001
```
Training set 
```
train_df['category'].value_counts().plot.bar()
```

Validation set 
```
validate_df['category'].value_counts().plot.bar()
```


###### Define terms 
```
batch_size=50 
FAST_RUN = False
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
```

## Define callbacks and learning rate 
A large learning rate allows the model to learn faster, at the cost of arriving on a sub-optimal final set of weights. A smaller learning rate may allow the model to learn a more optimal or even globally optimal set of weights but may take significantly longer to train.
To prevent overfitting, the learning will stop after 10 epochs and val_loss value not decreased. 
After trial and error, I decided to go with learning rate of 0.001. 
```
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystop = EarlyStopping(patience = 10)
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc',patience = 2,verbose = 1,factor = 0.5,min_lr = 0.001)
callbacks = [earlystop,learning_rate_reduction]
```


## Model building 

#### 1. Convolutional layer 
I believe that convolutional layers are filters, which are the core building blocks of convolutional neural networks. Different filters extract different parts of an image (example vertical lines or horizontal lines). More layers will help the model learn more complicated features. 
I used *conv2D()* function from Keras to build my first convolutional layer. The number of feature detectors/filters is set to be 32, and each filter dimension is (3,3). 
*input_shape* is the shape of input images on which we apply feature detectors through convolution. It is set to (128, 128, 3). 3 is the number of channels for a colored image, (128, 128) is the image dimension for each channel. 
The last argument is the activation function, which is set to *ReLU* to remove any negative pixel values in feature map. This will add non-linearity to a non-linear classification problem. 

#### 2. Batch normalization 
Batch normalization layer is a method to make the network faster and more stable through normalization of the input layer by recentering and rescaling. This has the effect of stabilizing the learning process and dramatically reducing the number of training epochs required to train deep networks. 

#### 3. Pooling layers 
Pooling layers reduce the input sizes of an image by downsampling to reduce the number of parameters and computation in the network. This is done without losing key features and spatial structure information in the images. 
A pooling layer follows after each convolutional layer, which performs a MAX operation which means it selects the maximum value inside each 2 x 2 matrix since I chose *pool_size=(2,2)*.

#### 4. Dropout layer 
A dropout layer drops some of the neurons as a form of regularization to prevent overfitting. 

#### 5. Flattening 
Finally, a flatten layer helps to pass output into a regular MLP. 

#### 6. Full connected layer 
With the above layers, we converted an image into a one-dimensional vector. Now, we will create a hidden layer. *output_dim* is the number of nodes in hidden layer. I used 512 and *ReLU* as activation function. 

#### 7. Output layer 
Then, I added an output layer. For binary classification, *output_dim* is 1, and activation functioon is *Sigmoid*. 

```
model=Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
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

## Model Compilation 
After all layers are added, I compiled the CNN using categorical classification. 
```
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics =['accuracy'])
```


## Model fitting 
I am aware that we use image augmentation (rotating, flipping etc) to increase the number of images for the training set to prevent overfitting. This splits images into different batches, and each batch will be applied random image transformation on a random selection of images to create many more images. 
I use flow_from_directory method to load images and apply image augmentation. The output of these generators will yield batches of 128x128 RGB images, each batch will have 50 samples. 
We will fit the generator to the model with 30 epochs, *validatioin_steps=5001/50=101*, *steps_per_epoch=19999/50 batches=400* . 

```
# training generator 
train_datagen = ImageDataGenerator(rotation_range=15,rescale=1./255,shear_range=0.1,zoom_range=0.2,     
    horizontal_flip=True,width_shift_range=0.1,height_shift_range=0.1 )
train_generator = train_datagen.flow_from_dataframe(train_df, 
    "/Users/admin/Documents/cat dog/dogsvscats/train", x_col='filename',y_col='category',
    target_size=IMAGE_SIZE,class_mode='categorical',batch_size=batch_size)

# validation generator 
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(validate_df, 
    "/Users/admin/Documents/cat dog/dogsvscats/train",x_col='filename',y_col='category',
    target_size=IMAGE_SIZE,class_mode='categorical',batch_size=batch_size)

# test generator 
test_datagen = ImageDataGenerator(rotation_range=15,rescale=1./255,shear_range=0.1,
    zoom_range=0.2,horizontal_flip=True,width_shift_range=0.1,height_shift_range=0.1)
```
> Found 19999 validated image filenames belonging to 2 classes.
> Found 5001 validated image filenames belonging to 2 classes. 

```
epochs=30
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=101,
    steps_per_epoch=400,
    callbacks=callbacks)
```


Now, let's fit and test the model. I obtained a training accuracy of __ and test accuracy of __ . The scores are reasonably close, suggesting the model is probably not over or underfit.


## Model improvements 
We should always strive to improve accuracy and reduce overfitting. This is by either adding more convolution layers or adding more dense layers.

