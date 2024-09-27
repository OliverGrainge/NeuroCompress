When using the ste estimator to perform quantized aware training, large quantization errors create noisy gradients and 
hence should be avoided. To do so, one can either increase the number of available quantization levels by increasing, 
bit-width, or train the network to converge to a solution around the quantization levels. The latter option is more 
attractive as it preserves the ability to use exteremely low bit-widths saving memory and or computation. 

To train the network to quantize to ternary values with little error, we must encourage the floating point weights to display 
a distribution equal to a gaussian mixture model with three peaks at $-\alpha$, $\alpha$ and $0$ where $\alpha$ is the scale parameter 
which is ideally minimized and $\sigma$ is also as small as possible. 

$p(w) = \pi_1 \mathcal{N}(\alpha, \sigma) + \pi_2 \mathcal{N}(-\alpha, \sigma) + \pi_3 \mathcal{N}(0, \sigma)$


To achieve such a distribution while additionally maximizing the task loss, one can apply a reguraizer the penalizes weights
with a distribution that deviates largely from this form. 

BitNet1.58 achieves this by computing the scale as the absolute mean of the weights. For a gaussian mixture of this form 
such a computation computes a scale below alpha. For instance, if you assume that there are each gaussian is equally probabale 
the mean of the gaussian mixture is. 

$\mu = \pi_1 * \alpha - \phi_2 * \alpha + \phi_2 * \alpha$ 

given that; 

$E[p(w)] = \pi_1 E[\mathcal{N}(\alpha, \sigma)] + \pi_2 E[\mathcal{N}(-\alpha, \sigma)] + \pi_3 E[\mathcal{N}(0, \sigma)]$

The absolute mean can be calculated by;



$E[|p(w)|] = \pi_1 E[|\mathcal{N}(\alpha, \sigma)|] + \pi_2 E[|\mathcal{N}(-\alpha, \sigma)|] + \pi_3 E[|\mathcal{N}(0, \sigma)|]$


For a normal distribution with mean of 0. The absolute mean can be computed as 

$E_{0} = E[|\mathcal{N}(0, \sigma)|] = \sigma \sqrt{\frac{2}{\pi}}$

for a non zero mean it can be given by; 

$E_{\alpha} = E[|\mathcal{N}(0, \sigma)|] = \sigma \sqrt{\frac{2}{\pi}} e ^ {\frac{\alpha^2}{2*\sigma^2}} + \alpha \left[1 - 2\Phi(-\frac{\alpha}{\sigma}\right)]$

Therefore now combining as a mixture 

$E[|w|] = \pi_1 E_{\alpha} + \pi_2 E_{-\alpha} + \pi_3 E_{0}$

produces a relationship between $\alpha$ and $\mu$ where $\mu$ is a fraction of what $\alpha$ should be leading to higher quantization errors. 


To overcome this issue, we allow the model to learn it's own scale, and in order to encorage it find a solution with 
low quantization error thereby reducing the noise in the gradients we add an additional regularization loss centered by the learned scale. Furthermore we, perform this regularization and quantization progressively to ensure a good signal early on in training 
allowing fast convergence and smoothing the transition from pre-training to quantization aware training. 

We initialize the floating point weights of the network with a uniform distribution in the range 

$[-b, b]$ where $b$ is the 1/in_features. We initialize the bias to zeros and the quantization scale to b/2 e.g. (E[|w|])

During training we progressivly quantize the network according to a sigmoid schedule with at each step.

$w_t = (1-\lambda) * w + \lambda * weight_quant(w)$ 
and with the same for activations except they are quantized to 8-bits 
$x_t = (1-\labmda_) * w + \lambda * weight_quant(x)$

For the regularization we apply the following loss at every step 

$l_{reg} = \frac{w}{\alpha} 


