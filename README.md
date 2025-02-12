# logistigate
python implementation of logistigate

https://logistigate.readthedocs.io/en/main/

Overview of logistigate
-----------------------
The logistigate methods analyze rates of substandard and
falsified products (SFPs) at locations within two echelons of a
larger, more complex supply chain, using testing data only from 
samples of products at locations of the lower echelon.

## Inference of SFP rates

Each location of both echelons has a fixed SFP rate.
Non-SFP products traveling through a location become SFP
according to this rate.
All products travel through one location of each echelon.
The probability of a lower-echelon
location obtaining product from upper-echelon locations is
stored in a "sourcing-probability matrix" for that supply
chain.
Product testing at the lower echelon is conducted with
a diagnostic tool with a known sensitivity and specificity.
Detection of an SFP is recorded as "1" and no detection is
recorded as "0."
There are two types of supply-chain information settings available
to regulators, Tracked and Untracked:

* In the Tracked case, both the upper-echelon and lower-echelon locations traversed by a product are known upon testing.

* In the Untracked case, the lower-echelon location where the product is obtained is known, as well as the system's sourcing-probability matrix.

The paper proposing these methods is available in IISE Transactions at
https://www.tandfonline.com/doi/full/10.1080/24725854.2023.2174277 or on Arxiv at
https://arxiv.org/abs/2207.05671
.

## Measuring the utility of sampling plans
Ensuring product quality is critical to combating the global challenge of substandard and falsified medical products. 
Post-marketing surveillance is a central quality-assurance activity in which products from consumer-facing locations 
are collected and tested. 
Regulators in low-resource settings use post-marketing surveillance to evaluate product quality across locations and 
determine corrective actions. 
The sampling plan in this surveillance which specifies where to test and the number of tests to conduct at a location.
We propose a Bayesian approach to generate a comprehensive utility metric for sampling plans. 
This sampling plan utility integrates regulatory risk assessments with prior testing data, available supply-chain 
information, and valuations of regulatory objectives.
We develop an efficient method for calculating sampling plan utility.

The paper proposing these methods is available on Arxiv at
https://arxiv.org/abs/2312.05678
.


## Funding
This work was funded through two National Science Foundation grants: EAGER Award 1842369 and NSF 1953111.