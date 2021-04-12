# Cats and dogs classification using CNNs 

Welcome! This is my first deep learning project and it is going to be a simple kernel on image recognition. Feel free to comment and make any changes, I am open to any feedback :) 

## 1. Problem statement: 
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


## 2. Define callbacks and learning rate 
A large learning rate allows the model to learn faster, at the cost of arriving on a sub-optimal final set of weights. A smaller learning rate may allow the model to learn a more optimal or even globally optimal set of weights but may take significantly longer to train.
To prevent overfitting, the learning will stop after 10 epochs and val_loss value not decreased. 
After trial and error, I decided to go with learning rate of 0.001. 
```
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystop = EarlyStopping(patience = 10)
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc',patience = 2,verbose = 1,factor = 0.5,min_lr = 0.001)
callbacks = [earlystop,learning_rate_reduction]
```


## 3. Model building 

#### Convolutional layer 
I believe that convolutional layers are filters, which are the core building blocks of convolutional neural networks. Different filters extract different parts of an image (example vertical lines or horizontal lines). More layers will help the model learn more complicated features. 
I used *conv2D()* function from Keras to build my first convolutional layer. The number of feature detectors/filters is set to be 32, and each filter dimension is (3,3). 
*input_shape* is the shape of input images on which we apply feature detectors through convolution. It is set to (128, 128, 3). 3 is the number of channels for a colored image, (128, 128) is the image dimension for each channel. 
The last argument is the activation function, which is set to *ReLU* to remove any negative pixel values in feature map. This will add non-linearity to a non-linear classification problem. 

#### Batch normalization 
Batch normalization layer is a method to make the network faster and more stable through normalization of the input layer by recentering and rescaling. This has the effect of stabilizing the learning process and dramatically reducing the number of training epochs required to train deep networks. 

#### Pooling layers 
Pooling layers reduce the input sizes of an image by downsampling to reduce the number of parameters and computation in the network. This is done without losing key features and spatial structure information in the images. 
A pooling layer follows after each convolutional layer, which performs a MAX operation which means it selects the maximum value inside each 2 x 2 matrix since I chose *pool_size=(2,2)*.

#### Dropout layer 
A dropout layer drops some of the neurons as a form of regularization to prevent overfitting. 

#### Flattening 
Finally, a flatten layer helps to pass output into a regular MLP. 

#### Full connected layer 
With the above layers, we converted an image into a one-dimensional vector. Now, we will create a hidden layer. *output_dim* is the number of nodes in hidden layer. I used 512 and *ReLU* as activation function. 

#### Output layer 
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

## 4. Model Compilation 
After all layers are added, I compiled the CNN using categorical classification. 
```
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics =['accuracy'])
```


## 5. Model fitting 
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
> Epoch 1/30
400/400 [==============================] - 291s 725ms/step - loss: 0.8854 - accuracy: 0.6393 - val_loss: 0.7012 - val_accuracy: 0.5757
WARNING:tensorflow:Learning rate reduction is conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy,lr
> Epoch 2/30
400/400 [==============================] - 289s 721ms/step - loss: 0.5089 - accuracy: 0.7529 - val_loss: 0.4699 - val_accuracy: 0.7698
WARNING:tensorflow:Learning rate reduction is conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy,lr
> Epoch 3/30
400/400 [==============================] - 300s 750ms/step - loss: 0.4194 - accuracy: 0.8108 - val_loss: 0.4135 - val_accuracy: 0.8206
WARNING:tensorflow:Learning rate reduction is conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy,lr
Epoch 4/30
400/400 [==============================] - 1624s 4s/step - loss: 0.3566 - accuracy: 0.8442 - val_loss: 0.3380 - val_accuracy: 0.8512
WARNING:tensorflow:Learning rate reduction is conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy,lr
Epoch 5/30
400/400 [==============================] - 257s 642ms/step - loss: 0.3198 - accuracy: 0.8594 - val_loss: 0.3786 - val_accuracy: 0.8294
WARNING:tensorflow:Learning rate reduction is conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy,lr
Epoch 6/30
400/400 [==============================] - 257s 642ms/step - loss: 0.2883 - accuracy: 0.8783 - val_loss: 0.4312 - val_accuracy: 0.8160
WARNING:tensorflow:Learning rate reduction is conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy,lr
Epoch 7/30
400/400 [==============================] - 260s 649ms/step - loss: 0.2780 - accuracy: 0.8840 - val_loss: 0.6568 - val_accuracy: 0.7888
WARNING:tensorflow:Learning rate reduction is conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy,lr
Epoch 8/30
400/400 [==============================] - 269s 673ms/step - loss: 0.2607 - accuracy: 0.8898 - val_loss: 0.3770 - val_accuracy: 0.8290
WARNING:tensorflow:Learning rate reduction is conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy,lr
Epoch 9/30
400/400 [==============================] - 273s 682ms/step - loss: 0.2352 - accuracy: 0.9019 - val_loss: 0.3609 - val_accuracy: 0.8690
WARNING:tensorflow:Learning rate reduction is conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy,lr
Epoch 10/30
400/400 [==============================] - 280s 699ms/step - loss: 0.2338 - accuracy: 0.9043 - val_loss: 0.3437 - val_accuracy: 0.8704
WARNING:tensorflow:Learning rate reduction is conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy,lr
Epoch 11/30
400/400 [==============================] - 283s 707ms/step - loss: 0.2241 - accuracy: 0.9066 - val_loss: 0.2582 - val_accuracy: 0.8888
WARNING:tensorflow:Learning rate reduction is conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy,lr
Epoch 12/30
400/400 [==============================] - 278s 694ms/step - loss: 0.2217 - accuracy: 0.9088 - val_loss: 0.3852 - val_accuracy: 0.8214
WARNING:tensorflow:Learning rate reduction is conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy,lr
Epoch 13/30
400/400 [==============================] - 292s 729ms/step - loss: 0.2206 - accuracy: 0.9092 - val_loss: 0.2819 - val_accuracy: 0.8780
WARNING:tensorflow:Learning rate reduction is conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy,lr
Epoch 14/30
400/400 [==============================] - 282s 704ms/step - loss: 0.2139 - accuracy: 0.9115 - val_loss: 0.2871 - val_accuracy: 0.8922
WARNING:tensorflow:Learning rate reduction is conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy,lr
Epoch 15/30
400/400 [==============================] - 291s 728ms/step - loss: 0.2029 - accuracy: 0.9150 - val_loss: 0.3894 - val_accuracy: 0.8908
WARNING:tensorflow:Learning rate reduction is conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy,lr
Epoch 16/30
400/400 [==============================] - 283s 708ms/step - loss: 0.2100 - accuracy: 0.9123 - val_loss: 0.4686 - val_accuracy: 0.7459
WARNING:tensorflow:Learning rate reduction is conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy,lr
Epoch 17/30
400/400 [==============================] - 285s 711ms/step - loss: 0.1876 - accuracy: 0.9237 - val_loss: 0.8282 - val_accuracy: 0.8088
WARNING:tensorflow:Learning rate reduction is conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy,lr
Epoch 18/30
400/400 [==============================] - 288s 719ms/step - loss: 0.1901 - accuracy: 0.9223 - val_loss: 0.8851 - val_accuracy: 0.7473
WARNING:tensorflow:Learning rate reduction is conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy,lr
Epoch 19/30
400/400 [==============================] - 279s 698ms/step - loss: 0.1870 - accuracy: 0.9271 - val_loss: 0.2697 - val_accuracy: 0.9158
WARNING:tensorflow:Learning rate reduction is conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy,lr
Epoch 20/30
400/400 [==============================] - 281s 701ms/step - loss: 0.1791 - accuracy: 0.9248 - val_loss: 0.2536 - val_accuracy: 0.9090
WARNING:tensorflow:Learning rate reduction is conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy,lr
Epoch 21/30
400/400 [==============================] - 1220s 3s/step - loss: 0.1731 - accuracy: 0.9303 - val_loss: 0.7398 - val_accuracy: 0.8408
WARNING:tensorflow:Learning rate reduction is conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy,lr
Epoch 22/30
400/400 [==============================] - 259s 646ms/step - loss: 0.1733 - accuracy: 0.9300 - val_loss: 0.3632 - val_accuracy: 0.8232
WARNING:tensorflow:Learning rate reduction is conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy,lr
Epoch 23/30
400/400 [==============================] - 324s 809ms/step - loss: 0.1784 - accuracy: 0.9276 - val_loss: 0.3574 - val_accuracy: 0.8906
WARNING:tensorflow:Learning rate reduction is conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy,lr
Epoch 24/30
400/400 [==============================] - 275s 686ms/step - loss: 0.1775 - accuracy: 0.9287 - val_loss: 0.2218 - val_accuracy: 0.9158
WARNING:tensorflow:Learning rate reduction is conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy,lr
Epoch 25/30
400/400 [==============================] - 278s 695ms/step - loss: 0.1670 - accuracy: 0.9336 - val_loss: 0.2578 - val_accuracy: 0.8934
WARNING:tensorflow:Learning rate reduction is conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy,lr
Epoch 26/30
400/400 [==============================] - 2075s 5s/step - loss: 0.1756 - accuracy: 0.9286 - val_loss: 0.3448 - val_accuracy: 0.8310
WARNING:tensorflow:Learning rate reduction is conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy,lr
Epoch 27/30
400/400 [==============================] - 4853s 12s/step - loss: 0.1666 - accuracy: 0.9340 - val_loss: 0.6403 - val_accuracy: 0.8050
WARNING:tensorflow:Learning rate reduction is conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy,lr
Epoch 28/30
400/400 [==============================] - 259s 646ms/step - loss: 0.1594 - accuracy: 0.9375 - val_loss: 0.2983 - val_accuracy: 0.9250
WARNING:tensorflow:Learning rate reduction is conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy,lr
Epoch 29/30
400/400 [==============================] - 258s 645ms/step - loss: 0.1706 - accuracy: 0.9340 - val_loss: 0.2862 - val_accuracy: 0.8826
WARNING:tensorflow:Learning rate reduction is conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy,lr
Epoch 30/30
400/400 [==============================] - 258s 645ms/step - loss: 0.1588 - accuracy: 0.9374 - val_loss: 0.2727 - val_accuracy: 0.8856
WARNING:tensorflow:Learning rate reduction is conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy,lr

We achieved an accuracy of 93.7% with validation accuracy of 88.6%. This is a fairly good result! 

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


## 6. Model improvements 
We should always strive to improve accuracy and reduce overfitting. This is by either adding more convolution layers or adding more dense layers.
I have decided to add another convolution layer into my model building and re-run steps 3 to 6. 
```
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
```
After fitting the model, we achieved 95.6% accuracy, validation accuracy 90.2%, loss of 11% and validation loss of 27.3%. 


## 7. Data augmentation 
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

