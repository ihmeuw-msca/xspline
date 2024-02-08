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
    affiliation: 1
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

A wide range of tools and packages exist to support modeling with splines. These tools include Splipy[https://pypi.org/project/Splipy/] [@johannessen2020splipy], splines [https://pypi.org/project/splines/], and spline support in scipy [https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html]. Despite these tools, several important gaps remain in python packages for spline modeling.  

The `xspline` package closes some of these gaps.  Xspline provides a simple and effective package for basis splines with unique functionality not afforded by other packages, including
 - customized extrapolation, using polynomials of any degree up to that of the original spline (e.g. linear, quadratic, and cubic for a cubic spline)
 - handling of derivatives and definite integrals as functions of spline coefficients, rather than numerical evaluations
This flexibility is not afforded by any available package, and taken together, make `xspline` useful and adaptable to many fields. In particular, xspline is regulary used in global health to fit, interpolate, and extrapolate nonlinear signals in a wide array of contexts in global health [@murray2020global]. Handling  


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
