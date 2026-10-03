# Declarations (paste into the submission system)

## Data availability
The ward-level feature table, labels, model predictions and the human-audit sheet generated in this study are available in the project repository: https://github.com/fmad121581-hub/informal-settlement-classifier. The underlying satellite and ancillary datasets are public and were obtained from their original providers: Landsat 9 (USGS), Sentinel-2 (Copernicus), ESA WorldCover 2021, SRTM DEM (NASA), WorldPop (2020), GADM level-4 boundaries, OpenStreetMap buildings (Geofabrik extract), and the EO4SD-Urban 2017 informal-settlement inventory (ESA). Large raw rasters are not stored in the repository; the scripts in `notebooks/` regenerate every derived file from them.

## Code availability
All processing, modelling and validation scripts (notebooks 01 to 13) are in the same repository, with `requirements.txt` and a fixed random seed (42). Running the scripts in the order given in the README reproduces every number in the manuscript. A permanent archive (for example a Zenodo DOI) can be added at acceptance.

## Declaration of competing interest
The author declares that he has no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Funding
This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors. [Confirm before submitting.]

## CRediT author statement
Fahim Ahmed: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing (original draft, review and editing), Visualization.

## Ethics
The study uses only satellite imagery, aggregated population rasters and administrative boundaries. No human participants or personal data were involved.

## Declaration of generative AI use
[Elsevier asks for this if AI tools were used. Edit to match what is true.] During the preparation of this work the author used Claude (Anthropic) to assist with code development and to improve the language and structure of the text. The author reviewed and edited all output and takes full responsibility for the content of the publication.
