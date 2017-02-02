# Implementing Siamese Network using Tensorflow with MNIST

<p align="center"> <img src="./result.png" width="600"> </p>

I have been interested in Siamese network. To my understanding, it is one way of dealing with weakly supervised problems. Its beauty lies in its simple scheme. It seems Siamese networks (and Triplet network) have been popularly used in many applications such as face similarity and image matching. A [web page](http://andersbll.github.io/deeppy-website/examples/) motivates me to implement a similar Siamese network using Tensorflow.

These codes here embed hand-written digits into 2D space. A loss function controls the embedding to be closer for digits in the same class and further for digits in the different classes. I borrowed visualization part from the original source with a little modification.

I keep codes simple for my personal experiments (e.g., different architectures or loss functions).

* `run.py` : nothing but a wrapper for running.
* `inference.py` :  architecture and loss definition. you can modify as you want.
* `visualize.py` : for visualizing result.

You can simply run  :

```bash
$ python run.py
...
step 34750: loss 0.179
step 34760: loss 0.113
step 34770: loss 0.078
...
```

This will download and extract MNIST dataset (once downloaded, it will skip downloading next time). The result will look like the image on the top. It saves an intermediate model regularly (with name `model.ckpt`) while training.

When you run `run.py`, if the file exists, you will be asked if you simply want to load it. `yes` will load the model, and show embedding result. So you can see the resulting embedding anytime (by stopping training or with a separate cmd/shell while training).

```bash
$ python run.py
We found model.ckpt file. Do you want to load it [yes/no]? yes
```

Please let me know if there are mistakes or comments. Thanks!

Youngwook Paul Kwon