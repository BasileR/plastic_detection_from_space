# Project

In this git, you will find our codes and research in the field of marine plastic detection. We have manipulated spatial data and explored different AI techniques for detection.  

## Branch develop 

This branch groups together all the commented codes that have been developped and that can be used for plastic detection for sentinel-2 type images. 

### Notebook Explanations

Each notebook is self explanatory. A description is provided at the top of each notebook. Let us give a brief recap of their content here :

* Plastic Spectral Unmixing   : Algorithms using spectral unmixing techniques with K-means for clustering. 
* Test_hypothese_Alina        : Statistical testing for plastic detection
* Plastic supervised learning : Algorithms based on supervised learning more specifically KNN and Naive Bayes 
* Graphs 		                  : Codes for detection with graph theory
+ a report that explains our project with more details 

### How to use utils to run the notebooks

If not already installed, set up all the libraries needed.
For now : tifffile, numpy, matplotlib.pyplot, spectral, os, glob and cv2

run the following command :
```
pip install tifffile
```
or directly in python shell :
```python
!pip install tifffile
```
