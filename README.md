# Leap Labs Testing Task

**It is required to train a model that adds a small amount of noise to the images so that the classifier makes a mistake.**

We impliment the ideas from the paper *Szegedy, Christian, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. “Intriguing properties of neural networks.” arXiv preprint arXiv:1312.6199 (2013)*.

Our algorithm takes as input:
- *image_path* (str) - path for image
- *target_class* (int 0-999) - the class that we want the classifier to put the given image
- *lr* (float) (by default  fult 1e-2) - learning rate
- *epoch_n* (int) (by default  250) - epoch number
- *noise_coef* (int) (by default  1e6) - noise multiplier in loss function

An example of a launch might look like this:

???

The result is the demonstration of the initial image and class and the new image and class. 


**Remark:** It may happen that the algorithm does not reach the target class. In this case, you will see a warning: *"Warring! The target class is not reached, increase the number of epochs"*
