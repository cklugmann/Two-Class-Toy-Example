<style>
img[src*='#left'] {
    float: left;
}
img[src*='#right'] {
    float: right;
}
img[src*='#center'] {
    display: block;
    margin: auto;
}
</style>

From scratch implementation for learning a (linearly separable) two class problem.
This is part of an introductory example in my master thesis. More details can be found in the text.

## Dataset 
A sample consisting of two dimensional real-valued input variables representing points with coordinates within [0, 1].
Classes are assigned to the individual points depending on whether they lie above (label 1) or below (label 0) the straight line y=x.

Depending on whether `use_noisy_labels` is set to true or false, the real labels or a noisy variant of them will be used. The degree of noise is determined by the perpendicular distance of the points to the decision boundary.

## Model
Logistic regression is used to address the problem. By construction, in both cases (noisy or non-noisy labels) it should be possible to learn an *ideal* classifier. We used batch stochastic gradient descent here to iteratively adjust the weights. The batch size, learning rate and number of epochs can be adjusted in the script. Note that in the case of noisy labels, a smaller learning rate is generally needed for the method to converge.

## Intuition
In the second case, where the labels do not take discrete values, the ground-truth can be viewed as a kind of voting. Such a voting could arise, for example, if not a single 'expert' contributes a single label as a 'gold standard', but many annotators provide an answer. A value y=0.6 can then be interpreted as an answer in which 60% of the annotators agree that the object belongs to class 1.

## Visualization

### With 'hard' labels
![Loss](./images/loss.jpg#left)
![Contour](./images/contour.jpg#right)
![Surface](./images/surface.jpg#center)

### With smooth labels
![Loss](./images/loss_noisy.jpg#left)
![Contour](./images/contour_noisy.jpg#right)
![Surface](./images/surface_noisy.jpg#center)



