**Date:** [submission date]

To the Editors,
*Remote Sensing Applications: Society and Environment*

**Re: Submission of "Ward-Level Informal Settlement Classification in Dhaka"**

Dear Editors,

I am submitting the manuscript "Ward-Level Informal Settlement Classification in Dhaka" for consideration as an original research article in *Remote Sensing Applications: Society and Environment*.

Most machine-learning work on informal settlements maps them at pixel or building scale from very-high-resolution commercial imagery. That is precise, but the unit is finer than the one Dhaka's planning agencies (RAJUK, DNCC, DSCC) actually budget and report on, and the imagery is expensive to refresh. This paper asks a more modest and more usable question: how well can all 203 wards of the Dhaka Metropolitan Region be classified as formal or informal using only free data (Landsat 9, Sentinel-2, WorldPop, OpenStreetMap)?

Three things distinguish the work.

First, it reports a problem many label-scarce remote-sensing studies share and rarely test. Our rule-based training labels used some of the same variables as the model features, which inflated accuracy to 90.9%. We separate the signal from the artifact with a decoupled protocol (rule-independent features only, anchor-only labels, thana-grouped and spatial-block cross-validation). The honest figure is 85.3% ± 7.7% against a 64.1% majority baseline, and 79.4% on the strictest anchor-only test.

Second, validation does not rest on the training labels alone. Ward-level predictions correlate with an independent 2017 informal-settlement inventory (EO4SD-Urban) at Spearman r = 0.65 (n = 203), and a blind visual audit of 40 wards gave Cohen's kappa = 0.45. We report that moderate agreement as a limitation and not as a success, together with the direction of the disagreements.

Third, the pipeline is interpretable (feature ablation, SHAP, Bland-Altman agreement analysis) and fully reproducible from a fixed random seed. Code and derived ward-level data are public.

The study is a single-city case with a modest labeled sample (170 wards), and the manuscript says so plainly. I believe it suits the journal's readership because it targets operational, low-cost mapping for planners and offers an evaluation check that transfers to other cities.

This manuscript is original, has not been published, and is not under consideration elsewhere. I am the sole author, and there are no competing interests to declare. I would like to publish through the standard subscription route, with no open-access fee.

Thank you for your time and consideration.

Sincerely,

Fahim Ahmed
Department of Urban and Regional Planning (DURP)
Bangladesh University of Engineering and Technology (BUET), Dhaka, Bangladesh
Email: fmad121581@gmail.com

*Suggested reviewers (add 3 to 4 real names before submitting; I have not invented any):* [name, affiliation, email, one-line reason]
