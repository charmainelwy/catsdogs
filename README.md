# Cats and dogs classification using CNNs 

Welcome! This is my first deep learning project and it is going to be a simple kernel on image recognition. Feel free to comment and make any changes, I am open to any feedback :) 

## Problem statement: 
I will be implementing Convolution Neural Network (CNN) Classifier to predict the category of dog or cat. I used the Asirra dataset, which can be found here. https://www.kaggle.com/c/dogs-vs-cats
I will be using Python on VSC. 

#### Creating training data
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
```
```
np.shape(df)
```
> (25001, 2)

There are 25001 images in our train dataset. 12501 cats and 12500 dogs. Data seems well balanced to proceed. 
<img width="393" alt="Screenshot1" src="https://user-images.githubusercontent.com/61202712/114333550-2640dd80-9b7b-11eb-9abc-79d97afc1365.png">

#### Train validation split 
Now, we will create a new dataset containing 2 subsets, a training set with 10,000 samples of each class (20,000 in total) and a validation dataset with 5001 total.
```
train_df, validate_df = train_test_split(df, test_size = 0.2, random_state = 42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
total_train=train_df.shape[0]  # 20 000 
total_validate=validate_df.shape[0]  #5001
```
###### Training set 
```
train_df['category'].value_counts().plot.bar()
```
<img width="393" alt="Screenshot1" src="https://user-images.githubusercontent.com/61202712/114343885-6c08a080-9b91-11eb-8259-8938c6f9b7be.png">

###### Validation set 
```
validate_df['category'].value_counts().plot.bar()
```
<img width="377" alt="Screenshot 2021-04-12 at 1 17 49 PM" src="https://user-images.githubusercontent.com/61202712/114343928-7c208000-9b91-11eb-9a16-17b832ccfa66.png">

#### Define terms 
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
The output of these generators will yield batches of 128x128 RGB images, each batch will have 50 samples. We will fit the generator to the model with 30 epochs, *validatioin_steps=5001/50=101*, *steps_per_epoch=19999/50 batches=400* . 

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

```
plot_accuracy_and_loss(history)
```


```
# prepare testing data
test_filenames = os.listdir("/Users/admin/Documents/cat dog/dogsvscats/test1")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]

test_generator = test_datagen.flow_from_directory(
     test_df, x_col='filename', y_col=None, target_size=IMAGE_SIZE, batch_size=50, class_mode=None, 
     shuffle=False)

test_loss, test_acc = model.evaluate_generator(test_generator, steps=101)
print('test acc:', test_acc)
```

```
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))

test_df['category'] = np.argmax(predict, axis=-1) #pick category with highest probability with numpy avg max 

# convering predict cat into generator class. It is the classes that image generator map while converting data into computer vision
label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)

test_df['category'] = test_df['category'].replace({ 'dog': 1, 'cat': 0 })

test_df['category'].value_counts().plot.bar()
```

Now, let's fit and test the model. I obtained a training accuracy of __ and test accuracy of __ . The scores are reasonably close, suggesting the model is probably not over or underfit.


## Model improvements 
We should always strive to improve accuracy and reduce overfitting. This is by either adding more convolution layers or adding more dense layers.
I have decided to add another convolution layer into my model building and rerun.
```
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
```

The output i had was __


## Data augmentation 
Data augmentation means to increase the number of data by adding more data (rotating, flipping, shearing) to existing dataset to reduce overfitting during training. 
Generate a number of random transformations on an image and visualize what it looks like. 

```
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_cats_dir = os.path.join(train_df, 'cats')
fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]

img_path = fnames[4] # Choose one image to augment

img = image.load_img(img_path, target_size=(224, 224)) # load image and resize it

x = image.img_to_array(img) # Convert to a Numpy array with shape (224, 224, 3)

x = x.reshape((1,) + x.shape)

# Generates batches of randomly transformed images.
# Loops indefinitely, so you need to break once four images have been created
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()
```

#### Build a model with data augmentation (_x per epoch) 

