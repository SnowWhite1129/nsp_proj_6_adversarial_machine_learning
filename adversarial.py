import numpy as np
from keras.preprocessing import image
from keras.activations import relu, softmax
import keras.backend as K
from keras.models import load_model
import os, sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def generate_adversarial(img_path):
    model = load_model('model.h5')
    dim = (224, 224)
    img = image.load_img(img_path, target_size=dim)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Get current session (assuming tf backend)
    sess = K.get_session()
    # Initialize adversarial example with input image
    x_adv = x
    # Added noise
    x_noise = np.zeros_like(x)
    
    # Set variables
    epochs = 100
    epsilon = 0.01

    if 'dog' in img_path:
        target_class = 0 # cucumber
    else:
        target_class = 1

    prev_probs = []
    
    for i in range(epochs):
        # One hot encode the target class
        target = K.one_hot(target_class, 2)

        # Get the loss and gradient of the loss wrt the inputs
        loss = -1*K.categorical_crossentropy(target, model.output)
        grads = K.gradients(loss, model.input)

        # Get the sign of the gradient
        delta = K.sign(grads[0])
        x_noise = x_noise + delta

        # Perturb the image
        x_adv = x_adv + epsilon*delta

        # Get the new image and predictions
        x_adv = sess.run(x_adv, feed_dict={model.input:x})
        pred = model.predict(x_adv)

        top_inds = pred.argsort()[::-1][:5]
    
        cls_list = ['cat', 'dog']

        if pred[0][target_class] > 0.95:
            print(i, pred[0][target_class], cls_list[target_class])
            return x_adv

    return x_adv


test = os.listdir(sys.argv[1])
for testcase in test:
    img = generate_adversarial(sys.argv[1] + testcase)
    filename = testcase.split('.')
    filename = sys.argv[1] + filename[0] + '_adversarial.png'
    image.save_img(filename, img[0])
