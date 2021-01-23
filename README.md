# Computing Valid p-value for Optimal Changepoint by Selective Inference using Dynamic Programming (NeurIPS 2020)

This package implements a method to compute valid p-value for optimal changepoint by selective inference using dynamic programming. Based on the selective inference (SI) framework, we propose an exact (non-asymptotic) approach to compute valid p-values for testing the significance of the changepoints. Although it is well-known that SI has low statistical power because of over-conditioning, we address this disadvantage by introducing parametric programming techniques. We propose an efficient method to conduct SI with the minimum amount of conditioning, leading to high statistical power.

See the paper <https://arxiv.org/abs/2002.09132> for more details.


## Installation & Requirements

This package has the following requirements:

- [numpy](http://numpy.org)
- [matplotlib](https://matplotlib.org/)
- [mpmath](http://mpmath.org/)

We recommend to install or update anaconda to the latest version and use Python 3
(We used Python 3.7.6).

## Reproducibility

All the figure results are saved in folder "/figure results" and some results are shown on console.

The following commands are run from the terminal.

- FPR and Power comparison when K is fixed (Figure 4)
	- FPR
	```
	>> python ex1_fixed_k_fpr_optseg_si_optseg_si_oc.py
	```
	- Power
	```
	>> python ex1_fixed_k_power_optseg_si_optseg_si_oc.py
	```
	
- FPR and Power comparison when K is unknown (Figure 5)
	- FPR
	```
	>> python ex2_unknown_k_fpr_optseg_si_optseg_si_oc.py
	```
	- Power
	```
	>> python ex2_unknown_k_power_optseg_si_optseg_si_oc.py
	``` 
	
- Power demonstration of the proposed OptSeg-SI (Figure 6)
	```
	>> python ex3_power_demo_optseg_si_optseg_si_oc.py
	```
	
- Evaluation of computing time for the OptSeg-SI method (Figure 7)
	
	We note that this experiment may take time because we repeat the experiment many times for each value of N.

	First we need to set the value of N. Please set the value of N in {200, 400, 600, 800, 1000, 1200} in the file ex4_computing_time.py. Then, please run the following command

	```
	>> python ex4_computing_time.py
	```
	
	Since we have already got the results in advance, please run the following command to reproduce Figure 7 in the paper
	
	```
	>> python ex4_plot_result.py
	```


