![GSoC Logo](/images/GSoC.png)

# GSoC 2025 - Final Submission 

 

**Name**: Monica Paoletti (mpaolett@sissa.it)

 

**Organization**: [INCF](https://github.com/incf) 

 

**Mentors**: [Meysam Hashemi](https://github.com/mhashemi0873), [Katya Zossimova](https://github.com/katya-zossi/), [Daniele Marinazzo](https://github.com/danielemarinazzo) 

 

**Project**: Transitioning from Simulation-Based Inference to Automatic MCMC for Virtual Brain Inference 



**Repo**: [repo](https://github.com/monicapaoletti/GSoC_T3.3_inference/tree/main)

 

**Project description** 

This project replaces Simulation-Based Inference (SBI) with Markov Chain Monte Carlo (MCMC) methods to perform parameter estimation in Virtual Brain Models (VBMs). We focus on a whole-brain, connectome-based model implemented within the Virtual Brain Inference (VBI) framework, in which each brain region is represented as a neural mass unit following the Montbrió–Pazó–Roxin equations and assumptions. The regions are coupled through an additive input current in the membrane potential equation and by the structural connectivity, weighted by a network scaling parameter (G) that modulates the influence of the structural connectivity on whole-brain dynamics. The model incorporates a Balloon–Windkessel framework, enabling the forward simulation of the BOLD fMRI signal. Communication between different brain regions and their associated dynamics is characterized by static functional connectivity (FC), functional connectivity dynamics (FCD), and brain fluidity. The latter is a statistical measure (variance) of FCD which accounts for the extent and frequency of transitions between distinct brain network states, thereby reflecting the flexibility or dynamical adaptability of large-scale functional connectivity patterns over time. Such summary features depend critically on G. The primary focus of this project is the estimation of the network scaling parameter from summary features, emphasizing parameter recovery through MCMC techniques. The implementation leverages the JAX framework to enable Just in Time Compilation (JIT), automatic differentiation, and efficient parallel computation on GPUs. By embedding the model into probabilistic programming frameworks such as PyMC and NumPyro, the project aims to establish a principled, MCMC-based Bayesian inference workflow that improves identifiability, interpretability, accuracy, and scalability compared to deep learning–based SBI approaches. 

 

**Project Goals**

The main objectives of the project were to translate the Virtual Brain Model (VBM) from its existing Numba/CuPy implementation into JAX, thereby enabling automatic differentiation and leveraging Just-In-Time (JIT) compilation for efficient MCMC-based inference. The project also aimed to benchmark the JAX implementation against existing Simulation-Based Inference (SBI) approaches to evaluate improvements in both accuracy and computational performance. Further goals include exploring additional or extended MCMC algorithms, conducting comparative analyses across CPU and GPU architectures, developing reparameterization strategies to enhance convergence, and ultimately scaling the model to empirical neuroimaging data to assess its applicability to real-world inference problems. 

 

**Work Completed** 

The VB model components were successfully translated into JAX, enabling JIT compilation and GPU compatibility. Summary features, including FC, FCD, and fluidity, were implemented in JAX as well to ensure computational efficiency in the inference process. Benchmarking experiments compared the JAX implementation against the existing Numba-based simulation on both CPUs and GPUs. 
Parameter exploration was conducted through grid search using Kolmogorov–Smirnov (KS) and Kullback–Leibler (KL) divergence metrics, both for the FC and FCD measures, along with parameter sweeping techniques, as implemented in the ‘vbjax’ framework, which allows efficient handling of functional evaluations and efficient parallelization for loss minimization in an optimization-based framework. However, optimization-based inference turned out to be unreliable for some specific parameter values due to the bifurcation shown by the single-unit neural masses, which is captured by the fluidity measure. 
The JAX model was then integrated into probabilistic programming environments (PyMC and NumPyro) to enable MCMC inference. Given the complexity of the posterior distributions, the current inference workflow focused on PyMC using gradient-free algorithms. Several samplers were tested, including Slice Sampling, Metropolis, Differential Evolution Metropolis (DE-Metropolis and DE-Metropolis Z), Sequential Monte Carlo (SMC), and Approximate Bayesian Computation (ABC)-based SMC. At the current stage of the project, the algorithm that gives better estimates is the ABC-based SMC, a sampling algorithm that relies on an ABC approach approximating the likelihood via a distance kernel and an SMC approach, which consists of the subsequent evolution of a population of particles (parameter samples) through a sequence of intermediate distributions, from the prior to the posterior. However, we also observed that this algorithm's efficacy relies on the proper choice of acceptance parameters. 

**Future Work** 

Future developments for this work involve scaling the framework to whole brain networks and simulation times comparable to those of existing experimental data using high-performance computing (HPC) resources.  

In addition, future work will explore hybrid inference techniques such as the Stochastic Normalizing Flow (SNF) approach or the Adaptive Monte Carlo augmented with Normalizing Flows, which combine stochastic sampling (e.g., MCMC methods) with Normalizing Flows (NF) sampling or exploration to address challenges arising from the posterior’s multimodality and efficient sampling, while enhancing efficient exploration. Alternative approaches involve Likelihood Approximation Network (LAN) inference that relies on a likelihood approximation leveraging the advantages of automatic function vectorization in JAX and our current parameter sweeping infrastructure to generate training examples and enable fast exact Bayesian inference, perhaps combining this approach with an approximate SMC framework. Hopefully, these strategies will make it possible to work to perform gradient-based inference as well. 

Applying the framework to real neuroimaging datasets remains an important future milestone for validating its scalability and neuroscientific relevance. 

 

**Challenges Encountered** 

Several technical challenges were encountered during the project. JIT-compiling the JAX implementation required non-trivial adjustments to handle static variables and ensure compatibility with JAX’s functional programming model. A custom, immutable class structure was designed to maintain both computational efficiency and structural flexibility within the JAX framework. A stochastic differential equation solver based on the Heun method was implemented to enable noisy, step-by-step integration of neural dynamics. Synchronizing GPU-based computations required using a JAX method that ensures the correct execution order by controlling asynchronous dispatch. Additional methodological complexity arose from the need to balance gradient-based and gradient-free inference approaches, address parallelization issues, and implement a customized PyMC Op to interface the model with probabilistic programming workflows. Finally, scalability constraints prevented the application of the inference methods to large-scale empirical data within the current timeframe. Overcoming these challenges deepened my understanding of the JAX ecosystem, probabilistic programming, and Bayesian inference, while enhancing my programming skills and engagement with open-source scientific software development. 

**Knowledge Gained** 

Through this project, I deepened my understanding of computational neuroscience in the context of the Dynamic Causal Modelling (DCM) focusing on Virtual Brain models, particularly neural mass models such as the Montbrió–Pazó–Roxin, connectome-based whole-brain models such as the SBI-VBMs or the models implemented in the VBI for fMRI and EEG data, and the Balloon–Windkessel model for the modeling of BOLD fMRI signal. I acquired practical experience in implementing stochastic differential equation solvers in JAX and learned the principles and limitations of Simulation-Based Inference (SBI). I also gained proficiency in parameter sweeping techniques, probabilistic programming with NumPyro and PyMC, and the comparison between gradient-based and gradient-free, as well as likelihood-based and likelihood-free inference methods. Finally, I developed familiarity with advanced Bayesian inference techniques, including Sequential Monte Carlo (SMC) and Stochastic Normalizing Flows (SNF), Adaptive Monte Carlo methods with Normalizing Flow and Likelihood Network Approximation (LAN), which will inform the future extensions of this work. 

**Acknowledgments** 

First, I would like to express my sincere gratitude to my mentors, Meysam Hashemi, Daniele Marinazzo, and Katya Zossimova, for their invaluable guidance, patience, and expertise throughout this project. Their mentorship has been instrumental in shaping both the technical and conceptual direction of my work. I would also like to thank the INCF for hosting this project and EBRAINS for providing the collaborative framework in which this work was developed. I am grateful for the opportunity to present my results as a Google Summer of Code (GSoC) contributor at the upcoming EBRAINS Summit in December. Finally, I extend my appreciation to Google for supporting the GSoC program and offering such an inspiring and impactful platform for student developers worldwide. I look forward to continuing my collaboration with my mentors and the INCF community and contributing to the future development of this research project. 

 