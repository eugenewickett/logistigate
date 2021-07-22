Overview of logistigate
-----------------------
The logistigate methods infer aberration likelihoods at
entities within a two-echelon supply chain, using
testing data only from samples of products at entities
of the lower echelon.

Each entity of both echelons has some fixed aberration
likelihood.
Products traveling through an entity become aberrational
according to this likelihood.
All products travel through one entity of each echelon.
The probability of a lower-echelon
entity obtaining product from upper-echelon entities is stored in
what is called the "sourcing matrix" for that system.
Testing of products at the lower echelon is conducted with
a diagnostic tool with a known sensitivity and specificity.
This testing yields aberrational (recorded as "1") or acceptable ("0")
results.
Information-availability settings are distinguished by
two categories, Tracked and Untracked:

* In the Tracked case, both the upper-echelon and lower-echelon entities traversed by a product are known upon testing.

* In the Untracked case, the lower-echelon entity where the product is obtained is known, as well as the system's sourcing matrix.

The logistigate methods were developed with the intent of
inferring sources of substandard or falsified products within
a two-echelon pharmaceutical supply chain.
Documentation throughout this package uses different terms
interchangeably.
In the language of this setting, aberrations are referred to
as substandard and falsified products, or SFPs.
In addition, entities of the upper echelon are referred to
as importers, and entities of the lower echelon are referred
to as outlets.