Overview of logistigate
-----------------------
The logistigate methods infer rates of substandard and
falsified products (SFPs) at locations within two echelons of a
supply chain, using testing data only from samples of products
at locations of the lower echelon.

Each location of both echelons has some fixed SFP rate.
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

The paper proposing these methods is available on Arxiv at
https://arxiv.org/abs/2207.05671
.

This work was funded through two National Science Foundation grants: EAGER Award 1842369 and NSF 1953111.