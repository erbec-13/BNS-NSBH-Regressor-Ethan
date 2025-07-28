# Regressor
The Random Forest regressor aims to predict the anticipated magnitude of kilonova events (BNS and NSBH) with an associated margin of error.

For features, the data will encompass aspects like the false alarm rate (related to detector statistics), signal-to-noise ratio, area(90), distance, longitude, latitude, HasNS, HasRemnant, HasMassGap, and P-astro.

The outcome or target data will capture the peak magnitude, which might fluctuate across filters. These filters, including G, R, and I, are pertinent to the ZTF (Zwicky Transient Facility) and also encompass u, y, z, K, H, and J.

# Data

Injections File: We have .dat files corresponding to O4 and O5. These files provide details about the masses and spins of BNS and NSBH events, along with their localization data (including longitude, latitude, and inclination angle) and distance. O4 denotes the current observing runs of the LIGO detector, whereas O5 represents projected future data.

Light Curves: This data consists of magnitudes, pass bands (u, g, r, i, y, z, H, K, and J), and corresponding simulation IDs. These IDs are consistent with those from the injections, ensuring that each event with specific mass and spin properties can be tracked.

Runs: This pertains to the metadata for sky maps.

ZTF Detection: The data here showcases the simulation ID and whether the event was detected by ZTF. A value of '0' indicates no detection, and '1' means it was detected.

Event Characteristics: This section includes properties like HasNS, HasRemnant, and HasMassGap (with NS ranging between 3 and 5 solar masses). Essentially, this information provides insights into the likelihood of the merger, including a neutron star or a remnant that expels some form of matter.

# LSTM Model predicts light curves without using mass ejecta

# LSTM Model predicts light curves using mass ejecta
