Overview of logistigate
-----------------------
Generally speaking, the logistigate methods infer aberration likelihoods at
entities within a two-echelon supply chain, only using testing data from sample
points taken from entities of the lower echelon. It is assumed that products
originate within the system at one entity of the upper echelon, and are
procured by one entity of the lower echelon. The likelihood of a lower-echelon
entity obtaining product from each of the upper-echelon entities is stored in
what is deemed the "transition matrix" for that system. Testing of products at
the lower echelon yields aberrational (recorded as "1") or acceptable ("0")
results. We then distinguish possible information-availability settings into
two categories, Tracked and Untracked:

* In the Tracked case, both the upper-echelon and lower-echelon entities traversed by the tested product are known upon testing.

* In the Untracked case, only the lower-echelon entity is entirely known, in addition to the system's transition matrix.

It is further assumed that products are aberrational at their origin in the
upper echelon with some entity-specific fixed probability, and that products
acceptable at the upper echelon become aberrational at the destination in the
lower echelon with some other entity-specific fixed probabiltiy. It is these
fixed probabilities that the logistigate methods attempt to infer.

More specifically, the logistigate methods were developed with the intent of
inferring sources of substandard or falsified products within a pharmaceutical
supply chain. Entities of the upper echelon are referred to as importers, and
entities of the lower echelon are referred to as outlets. This terminology is
used interchangeably throughout the logistigate package.