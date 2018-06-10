# CNN_VGG19_CUDA

Convolutional Neural Network of VGG19 model using CUDA to accelerate

## CNN architecture

VGG19 ([imagenet-very-deep-vgg19.mat](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat)) - pretrained model by imagenet with 19 layers

## Compile

```
make
```

## Execution

```
./cnn_vgg19_cuda <image file in .txt> vgg19_weight.txt vgg19_bias.txt vgg19_output_1000.txt
```

## Some useful tools

tools written by myself that will help a lot 

### vgg19.py

analyze imagenet-very-deep-vgg19.mat(need to download by yourself) and output to vgg19_weight/bias.txt

```
make vgg
```

### image_converter.py

convert RGB value of .jpg(224 * 224) into .txt (in RGB order)

```
make image
```

### softmax.py

convert output of the model, vgg19_output.txt, into problilities of 1000 classes corresponding to synset_word.txt and write to vgg19_probs.txt

```
make softmax
```

## image folder        

contain some .jpg files and its corresponding .txt and predict files

### p.s it's not actually a trainable model, just a reconstruction of vgg19 in evaluation phase
