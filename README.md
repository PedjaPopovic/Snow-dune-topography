# Snow-dune-topography
Python code to generate the synthetic "snow dune" topography described in the paper "Snow topography on undeformed Arctic sea ice captured by an idealized "snow dune" model," submitted to JGR: Oceans. The code was tested on a 64-bit Mac using python 2.7.10. 

## Description

The code is contained within the file snow_dune_topo.py. This code is meant to help reproduce the results of the paper. It reproduces the properties of the synthetic topography discussed in the paper but does make the comparison with LiDAR measurements of the snow topography. The code first generates and plots the synthetic "snow dune" topography, represented as a sum of randomly placed mounds of Gaussian shape with varying radius and height. Then, it finds the moments, the correlation function, and the height distribution of this generated topography discussed in the paper. Next, it calculates the melt pond evolution on this topography in three different ways discussed in the paper (numerical 2d model, first-order analytical estimate, and second-order analytical estimate). Finally, it plots the melt pond development diagram that determines whether the ponds will develop during summer. 

## Files

snow_dune_topo.py
  - Code that generates the synthetic topography and reproduces plots shown in the paper. 

## Dependencies
NumPy, SciPy, Matplotlib 
