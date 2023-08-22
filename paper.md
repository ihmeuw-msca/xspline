---
title: 'pysr3: A Python Package for Sparse Relaxed Regularized Regression'

tags:
  - Python
  - Splines
  - Derivatives
  - Integrals 
  - Flexible Extraploation
  - Design matrix 

authors:
  - name: Peng Zheng
    orcid: 0000-0003-3313-215X
    affiliation: 2
  - name: Aleksandr Aravkin
    orcid: 0000-0002-1875-1801
    affiliation: "1, 2"

affiliations:
 - name: Department of Health Metrics Sciences, University of Washington
   index: 1
 - name: Department of Applied Mathematics, University of Washington
   index: 2

date: 08.21.2023
bibliography: paper.bib

---

# Summary

Splines are a fundamental tool for describing and estimating nonlinear relationships [@de1978practical]. They allow nonlinear functions to be represented as linear combinations of spline basis elements. Researchers in physical, biological, and health sciences rely on spline models in conjunction with statistical software packages to fit and describe a vast range of nonlinear relationships. 

Remarkably, despite decades of research and availability of a wide range of software tools, including Splipy[https://pypi.org/project/Splipy/] [@johannessen2020splipy], splines [https://pypi.org/project/splines/], and support in scipy [https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html] several important gaps remain in python support for spline modeling.  

We fill this gap by providing a simple and effective package for basis splines that provides unique functionality not afforded by any other packages, including
 - customized extrapolation, using polynomials of any degree
 - handling of derivatives and definite integrals as spline functions, rather than evaluations
This flexibility is not afforded by any available package, and taken together, make `xspline` useful and adaptable to many fields. In particular, xspline is regulary used in global health to fit, interpolate, and extrapolate nonlinear signals in a wide array of contexts [@murray2020global].  


# Statement of Need
   
- why customized extraplolation is needed. 
- what do packages currently provide 
- why derivatives and integrals are needed
- what current packages currently provide 



# Core idea and structure of `xpsline`

- Peng to write 

More information about the structure of the library can be found in [documentation](https://zhengp0.github.io/xspline/api_reference/), 
while the mathematical use cases are extensively discussed in [@zheng2021trimmed] in the context of fitting risks. 


# Ongoing Research and Dissemination

Cite more papers that use xspline. 

# References
