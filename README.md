# Practical

# Intro
Our goal is to learn a generative model which learns to go from pose-distribution to image-distribution (or in other words: to generate images of a scene from a given pose). Furthermore, we want to invert our generative model: Infer the pose for a given image / view.


# Data
We're going to work with very simple (sparse) data

lego https://drive.google.com/file/d/1Gb0AbE3KPkJYDJnNgYtZG58ntuJNpVjQ/view?usp=sharing

chair https://drive.google.com/file/d/1kZoZyUoizT8ICnWBKt6OpBWdFHXdoICt/view?usp=sharing

hotdog https://drive.google.com/file/d/1Chp2-2odW-leLXgF4_MG_7d69ZAJOKwN/view?usp=sharing


# TASKS
* Train a NeRF first on a given scene (with notebooks/nerf_train.ipynb)
* Using trained NeRF, generate data (images, poses). 
* Train our generative model (SiT)[https://github.com/willisma/SiT] on this generated dataset. Instead of gaussian noise as the starting point, we use our poses.
* After training: Can we invert the flow (from image -> pose)?

# Experiments
* What is the reconstruction quality of our model?
* How robust is the model? i.e. if we sample a pose far away from our dataset, is the generated view still plausible?
* If you have some cool ideas, test them :smile:

# Notes
## Mathamtical detials
In Flow matching we are learning the flow from our initial distribution $x_0 \sim p(x_0)$ to our target-distribution $x_1 \sim p(x_1) = p_\text{data}$
with the vector-field regression loss
$\mathcal{L}_{FM} = \mathbb{E}_{t, x_0, x_1}\| v_\theta(x_t, t, c) - (x_1 - x_0)\|_2$
for some conditioning signal $c$ and out interpolant $x_t$ with.
$x_t = t x_1 + (1-t) x_0$
with timestep $t \in [0,1]$. Time $t=1$ corresponds to our data, $t=0$ corresponds to samples from our initial distribution $p(x_0)$.

**Note**: we can go from $x_1$ to $x_0$ by using Euler backwards.
### How do we generate samples? 
Sine we learned a velocity-field we can pick a starting-point $x_0 \sim p_(x_0)$ and *simulate* the ODE, which means we integrate the velocity field $v_\theta$ over time with an ODE-solver (for example Euler)
## What we will be doing
In standard flow matching we have $p(x_0) = \mathcal{N}(0,I)$, i.e. we start form Gaussian Noise and learn to generate images by denoising them. However, Flow Matching allows that we can choose our $p(x_0)$. In this case, we choose $p_0(x_0)$ as the gaussians centered around our poses: $x_p = P + \varepsilon$, where $\varepsilon \sim \mathcal{N}(0,1)$ and $P$ is a pose, i.e. it is the rotation and translation of our camera $P = [R|t]$. You may need to find a few tricks to make the dimensionality work (our interpolant $x_t$ is defined in $\mathbb{R}^{32\times32\times4}$), which is larger than the shape of $P \in \mathbb{R}^{3\times4}$. **Inspiration/Hint**: You could either use Plücker rays or repeat $P$ until the dimension is correct. 
