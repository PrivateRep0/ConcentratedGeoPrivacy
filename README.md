# Concentrated Geo-Privacy
This repo contains the official code for the experiments in the paper "Concentrated Geo-Privacy".

## Requirements/Dependencies
The algorithms and tests were implemented in Python (v3.10).

The following packages can be installed via conda or pip. The specific versions used in the reported experiments are:<br/>
numpy v1.24.3 <br/>
scipy v1.10.1 <br/>
shapely v2.0.1 <br/>

## Evaluation
To reproduce the results in a specific figure, run its corresponding script listed below.
| File name          | Corresponding figure | 
| ------------------ |--------------------- |
| test_traj_rho.py       | Figure 2(a),(b).            |
| test_traj_m.py         | Figure 2(c),(d).            |
| test_kpnn_rho.py       | Figure 3.                   |
| test_kpnn_m.py         | Figure 4.                   |
| test_convh_rho.py      | Figure 5(a).                |
| test_convh_m.py        | Figure 5(b).                |

For example, to run the test in Figure 5(b):
```test
python test_convh_m.py
```
## Data Source
The dataset in the ./data/ folder is from: 
CRAWDAD dataset epfl/mobility (v. 2009-02-24). Downloaded from https://crawdad.org/epfl/mobility/20090224. https://doi.org/10.15783/C7J010

## Compute Resources
All experiments reported in the paper were run on a linux machine with 8 CPUs and 32GB RAM. <br/>
