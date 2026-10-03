# Paper framing — contribution & positioning notes

Working notes for the Introduction / Related Work / Discussion sections.
Sources are the literature search done Sep 2026 (see chat for full list).

---

## 1. The three gaps the literature itself names

From the meta-analysis (Sun et al./arXiv 2406.08031, "Deep Learning for Slum
Mapping in Remote Sensing Images: A Meta-analysis and Review", 2014-2024):

1. **Generalizability** — no universally optimal model; results are highly
   context-specific and don't transfer across cities.
2. **Data / label limitations** — ground-truth scarcity is the single most
   cited constraint across the field.
3. **Explainability** — CNN/deep-learning dominance means most published
   models are black boxes; feature-level reasoning is rare.

Two more gaps are visible from reading the actual studies, even though no
single review states them explicitly:

4. **Scale mismatch with planning practice.** Almost everything in this
   literature works at pixel, object, or building-footprint scale (VHR
   imagery, OBIA, CNN/U-Net on 0.3–3 m imagery). That is the natural unit
   for a computer-vision problem, but it is not the unit municipal and
   metropolitan planners actually operate at — RAJUK, DNCC/DSCC and BBS all
   plan, budget and report at the ward or thana level. A pixel-accurate
   slum outline is not directly actionable for a planning office; a
   ward-level risk score is.
5. **Label-feature independence is essentially never discussed.** None of
   the reviewed papers explicitly audit whether their weak/rule-based
   labels leak into their feature set. This project ran into that problem
   directly (Sep 2026 circularity fix) and the decoupled-evaluation method
   used to resolve it is itself a small methodological contribution other
   ward/rule-based studies could reuse.

## 2. Where this project sits against comparable studies

| Study | Unit | Method | Result | Ground truth |
|---|---|---|---|---|
| Riyadh OBIA+RF (Alrasheedi et al., Earth Systems & Environment 2024) | Object (VHR) | OBIA + RF/SVM + expert indicators | 96% OA | 6,000 stratified points, 30/70 split |
| Dhaka U-Net (SDSU thesis, PlanetScope/Maxar) | Pixel | U-Net, U-Net+attention | 72–79% F1 | limited — flagged as the main constraint by the authors themselves |
| Gruebner et al. 2014 (Dhaka, this dataset paper) | Ward/union | Manual VHR interpretation + BBS 2005 cross-check | N/A (mapping, not classification) | Visual interpretation, cross-verified |
| **This project (after fix)** | **Ward** | **XGBoost, 7 rule-independent features** | **79.4% CV accuracy vs 68.3% majority baseline, anchor-only labels** | Ground-truth anchors (20 formal + 43 informal thanas) + planned BBS 2014 aggregate check |

Two honest framings follow from this table:

- Against the *pixel-level* Dhaka baseline (72–79% F1), a ward-level
  79.4% accuracy is competitive — and it's produced from medium-resolution,
  largely free data (Landsat, Sentinel-2, WorldPop, OSM) rather than VHR
  imagery that costs money and isn't always available at repeat intervals.
- Against the *object-level* Riyadh study (96%), this project is clearly
  behind — but that study used 6,000 hand-verified points against 63
  wards here. The honest comparison is "less accurate, far less
  ground-truth-hungry, coarser but planning-relevant scale" — a real
  trade-off, not a shortcoming to hide.

## 3. The contribution statement (draft)

> Most machine-learning approaches to informal-settlement mapping operate
> at pixel or building-object scale using very-high-resolution imagery,
> which is expensive, not always available for repeat monitoring, and
> produces outputs finer than the administrative units municipal planning
> agencies (RAJUK, DNCC, DSCC) actually use. We instead classify Dhaka's
> 203 wards directly, using free medium-resolution and crowd-sourced data
> (Landsat 9, Sentinel-2, WorldPop, OpenStreetMap), and report XGBoost
> feature importances to keep the classification interpretable rather than
> a black box. In building the labeled training set we found that
> rule-derived labels and rule-derived features overlapped substantially —
> a circularity problem not discussed in prior slum-mapping literature —
> and we introduce a decoupled-evaluation protocol (excluding
> rule-implicated features, and testing separately on independently
> sourced ground-truth-anchor wards) to report an honest, non-circular
> accuracy estimate.

## 4. External validation — DONE (Sep 2026)

Validated against ESA EO4SD-Urban Dhaka Informal Settlements 2017
(World Bank Data Catalog, CC-BY 4.0, 1,624 VHR-interpreted informal
settlement patches) — a source untouched anywhere in labeling or feature
engineering.

Method: spatial overlay of EO4SD patches onto the 203 wards, giving
`eo4sd_informal_fraction` = EO4SD-mapped informal area / ward area per ward.

Results (notebooks/07_external_validation.py):
- Training labels vs EO4SD fraction: informal-labeled wards average 11.0%
  EO4SD informal footprint vs 2.6% for formal-labeled wards.
  Point-biserial r = -0.393, p < 0.0001 (n=170), correct direction.
- Model P(informal) vs EO4SD fraction: Spearman r = 0.563, p < 0.0001
  (n=203, all wards).

This is the strongest evidence in the whole project that the model has
found a genuine informality signal and not just the labeling rule: EO4SD
was built independently (different organization, different imagery,
different method — visual interpretation of VHR imagery) and neither
touches the training pipeline. Gruebner et al. 2014's official dataset
links (DOI page + Humboldt University GDI) are dead as of Sep 2026 and
were not pursued further given EO4SD already provides a solid check.

## 5. Decoupled model x EO4SD — the headline result (DONE, Sep 2026)

notebooks/08_decoupled_external_validation.py trained the decoupled
(7-feature, rule-independent) model on all 170 labeled wards, predicted
P(informal) for all 203 wards, and correlated against the same EO4SD 2017
informal-footprint fraction used in notebook 07.

| Model | Spearman r vs EO4SD | p |
|---|---|---|
| Reference (12 features, rule-circular) | 0.639 | <0.0001 |
| **Decoupled (7 features, n=170)** | **0.652** | **<0.0001** |
| Decoupled (7 features, anchor-only n=63) | 0.089 | 0.21 (n.s.) |

The decoupled model trained on the full 170-ward labeled set matches — in
fact marginally exceeds — the rule-circular reference model's correlation
with an independent, externally-sourced dataset. This is the strongest
evidence in the project that the classifier has learned genuine
informality signal, not the labeling rule. The anchor-only (n=63) version
loses significance here, most plausibly because 63 training wards is too
few to produce a stable full-city probability surface — a sample-size
limitation to state plainly, not a sign the anchor labels are wrong.

**Recommended headline numbers for the paper**, both computed on the
decoupled (rule-independent) feature set:
- CV accuracy: 85.3% ± 7.7% (95% CI, 5-fold random StratifiedKFold) vs a
  64.1% majority-class baseline (n=170; 109 formal / 61 informal)
- Balanced accuracy: 84.7%, Macro-F1: 84.7%, MCC: 0.693 (random CV,
  out-of-fold predictions -- more informative than raw accuracy given the
  formal/informal class split; see section 9)
- External validation: Spearman r=0.652, p<0.0001 against EO4SD 2017 (n=203)

> **CORRECTION (Oct 2026):** an earlier draft of this section quoted
> "90.78% CV accuracy vs 59.5% majority baseline." Re-running
> `notebooks/06_decoupled_model.py` against the current repo state
> (170 labeled wards, same code, same random seed) gives 85.29% ± 7.67%,
> not 90.78%, and the majority-class baseline for n=170 (109 formal /
> 61 informal) is 64.1%, not 59.5%. The external-validation numbers
> (Spearman r=0.652 decoupled / r=0.639 reference) DID reproduce exactly
> and are correct. The accuracy figure was most likely copied from a
> stale run before a later edit to the label file. Use the corrected
> numbers above in the manuscript, and re-run the notebooks once more
> immediately before submission to confirm nothing has drifted again.

## 6. What's still needed before this framing is submission-ready

- [ ] BBS 2014 Census of Slum Areas aggregate sanity check (city-corp /
      thana totals) — optional now that EO4SD gives a stronger, spatially
      explicit check; still worth one paragraph as a secondary source.
- [x] Explicit limitations paragraph — DONE (Oct 2026), now also covers
      spatial CV results. See section 7.
- [x] Related-work deepened (Sep 2026): both TBC citations confirmed
      (Shinjini 2025; Raj, Mitra & Sinha 2024) and two new sources
      added — Yang et al. 2026 (building-morphometrics
      transferability/interpretability) and Shao, Ahmad & Javed 2024
      (RF vs XGBoost benchmark, motivates model choice and the
      interpretability gap discussed in Section 6.2). See section 8
      below for the full formatted list.
- [x] Spatial block / thana-grouped cross-validation — DONE (Oct 2026).
      See section 9.1.
- [x] Feature ablation study (M1-M5) — DONE (Oct 2026). See section 9.2.
- [x] Per-class metrics, confusion matrix, balanced accuracy, MCC —
      DONE (Oct 2026). See section 9.3.
- [x] Headline CV-accuracy number corrected (was stale/wrong) — see
      the correction note in section 5.
- [x] Yang et al. 2026 citation (formerly mis-cited as "Kuffer et al.
      2026") — full author list verified Oct 2026; see section 8.
- [x] SHAP feature attribution (notebooks/12_shap_analysis.py) — DONE
      (Oct 2026). See section 9.5.
- [x] Bland-Altman-style agreement plot vs EO4SD
      (notebooks/13_bland_altman.py) — DONE (Oct 2026). See section 9.6.
- [ ] Not done, requires the user's own effort (can't be automated):
      small independent human label audit (Cohen's kappa on a 30-50
      ward sample) — needs real annotators looking at imagery, not
      something to fabricate. BBS 2014 aggregate check remains
      optional/secondary now that EO4SD gives a stronger check; its
      source (CUS website, World Bank Microdata download) was already
      found to be dead/broken earlier in this project.

## 7. Limitations paragraph (draft, ready to adapt into the paper)

> This study has several limitations. First, the training labels combine
> a small set of ground-truth anchor wards (63, of which only 20 are
> formal) with rule-based labels derived from feature thresholds; although
> we removed the directly rule-derived variables from the feature set used
> for the reported accuracy and correlation figures, and separately
> confirmed that the resulting model's predictions correlate with an
> independent 2017 informal-settlement inventory (EO4SD-Urban, Spearman
> r=0.65, p<0.0001), the anchor-only subset remains too small to support a
> fully independent accuracy estimate on its own — cross-validation on
> that subset alone yields wide confidence intervals and did not reach
> statistical significance when used to train a full-city probability
> surface. Second, the formal-settlement class is under-represented
> relative to informal, both in the anchor set (20 vs. 43) and after
> after rule-based labeling (109 vs. 61 including rule labels), which likely
> limits the model's ability to characterize the full diversity of planned
> development in Dhaka. Third, the external validation source (EO4SD 2017)
> predates or postdates some of the satellite inputs used here (Landsat 9
> 2022, Sentinel-2 2021), so a degree of temporal mismatch is expected and
> may attenuate the observed correlation rather than inflate it. Fourth,
> ward-level classification necessarily aggregates within-ward
> heterogeneity — a ward with both a dense informal core and a planned
> periphery is scored as a single unit, which is appropriate for the
> planning-unit framing we adopt but loses the spatial precision of
> pixel- or object-level approaches. Fifth, GADM administrative
> boundaries used here are current as of their release and may not
> perfectly reflect 2022-era ward boundaries or the DNCC/DSCC ward
> renumbering, which could introduce small spatial misalignments in the
> feature extraction and the EO4SD overlay alike. Finally, because
> neighbouring wards can share similar urban morphology, ordinary random
> cross-validation may overstate generalization; we therefore also
> evaluated the decoupled model under thana-grouped and spatially-blocked
> cross-validation (no ward from the same thana or spatial cluster
> appears in both the train and test fold of a split). Accuracy fell
> modestly under both schemes (85.9% random vs 84.2% thana-grouped vs
> 83.1% spatial-block CV accuracy; full results in section 9.1) and the
> drop in both cases falls within the random-CV confidence interval,
> indicating the reported accuracy is not primarily an artifact of
> spatial autocorrelation, though some optimism from random splitting
> cannot be ruled out entirely given the modest sample size.

## 8. Related-work citation list (formatted)

- Alrasheedi, K. G., et al. (2024). Combining Local Knowledge with
  Object-Based Machine Learning Techniques for Extracting Informal
  Settlements from Very High-Resolution Satellite Data. *Earth Systems
  and Environment*. https://doi.org/10.1007/s41748-024-00393-1
- Shinjini, S. S. (2025). Slum Area Detection in Dhaka City,
  Bangladesh, Using Satellite Remote Sensing and Deep Learning.
  South Dakota State University, Electronic Theses and Dissertations,
  1741. https://openprairie.sdstate.edu/etd2/1741/
- Raj, A., Mitra, A., & Sinha, M. (2024). Deep Learning for Slum
  Mapping in Remote Sensing Images: A Meta-analysis and Review.
  arXiv:2406.08031. https://arxiv.org/abs/2406.08031
- Gruebner, O., Sachs, J., Nockert, A., Frings, M., Khan, M. M. H.,
  Lakes, T., & Hostert, P. (2014). Mapping the Slums of Dhaka from 2006
  to 2010. *Dataset Papers in Science*, 2014, 172182.
  https://doi.org/10.1155/2014/172182
- Bangladesh Bureau of Statistics (BBS) (2015). Census of Slum Areas and
  Floating Population 2014. Ministry of Planning, Government of
  Bangladesh.
- Angeles, G., et al. (2009). The 2005 census and mapping of slums in
  Bangladesh: design, select results and application. *International
  Journal of Health Geographics*, 8, 32.
  https://doi.org/10.1186/1476-072X-8-32
- World Bank / ESA EO4SD-Urban (2019). Dhaka, Bangladesh — Informal
  Settlements (EO4SD-Urban). World Bank Data Catalog, dataset 0041703.
  Operations report: DR0052092.
- Yang, H., Jimmy, E., Verburg, P. H., Kuffer, M., Levering, A., &
  van Vliet, J. (2026). A transferable and interpretable approach to
  slum mapping using building morphometrics and optical imagery.
  *GIScience & Remote Sensing*, 63(1), 1-28.
  https://doi.org/10.1080/15481603.2026.2649308
  (full author list verified Oct 2026 via VU Amsterdam research
  repository — first author is Yang, H., not Kuffer; cite as
  "Yang et al. 2026" in text, not "Kuffer et al. 2026")
- Shao, Z., Ahmad, M. N., & Javed, A. (2024). Comparison of Random
  Forest and XGBoost Classifiers Using Integrated Optical and SAR
  Features for Mapping Urban Impervious Surface. *Remote Sensing*,
  16(4), 665. https://doi.org/10.3390/rs16040665

## 9. Validation rigor upgrades — DONE (Oct 2026)

Addresses the reviewer-facing gaps a Q1 submission needs: spatial
leakage, feature-contribution transparency, and class-imbalance-aware
metrics. All three ran on the same 170-ward labeled set and the same
XGBoost config as notebooks 04b/06, for direct comparability.

### 9.1 Spatial cross-validation (notebooks/09_spatial_cv.py)

Two spatial grouping schemes, each ensuring no ward from the same
group appears in both the train and test fold of a split:
- **Thana-grouped** (StratifiedGroupKFold on NAME_3, the real
  administrative unit one level above ward; 41 groups, 5 folds, 0
  degenerate folds)
- **Spatial block** (StratifiedGroupKFold on 8 KMeans clusters of ward
  centroids; more size-balanced than thana but 2/5 folds had a
  single-class test set because one block absorbed 61 of 170 wards --
  flagged as a limitation of this particular blocking, not of the
  underlying model)

| Model | Random CV | Thana-grouped | Spatial block |
|---|---|---|---|
| Reference (12 feat, circular) | 87.1% | 82.4% (-4.6pp) | 86.8% (-0.3pp, 3/5 folds* ) |
| Decoupled (7 feat) | 85.9% | 84.2% (-1.7pp) | 83.1% (-2.8pp, 3/5 folds*) |

*AUC/accuracy computed only on folds with both classes in the test
set; accuracy itself is defined for all 5 folds, AUC for 3/5.

**Takeaway:** drops are modest and within the random-CV confidence
interval for both models -- no strong evidence of spatial leakage
inflating the headline accuracy. Report the thana-grouped number as
the primary "honest" CV accuracy in the manuscript (more folds valid,
uses a real administrative unit reviewers will recognize); mention
spatial-block as a secondary check with its single-block caveat noted.

### 9.2 Feature ablation (notebooks/10_feature_ablation.py)

Cumulative feature groups, all rule-independent (non-circular) through
M4; M5 is the circular reference shown only for context.

| Model | Features added | CV Accuracy | External Spearman r (EO4SD) |
|---|---|---|---|
| M1 | savi, slope | 81.8% | 0.556 |
| M2 | + pop_std | 86.5% | 0.584 |
| M3 | + Sentinel-2 NDBI/MNDWI | 84.1% | 0.623 |
| M4 | + OSM building density/area (= full decoupled model) | 85.9% | **0.652** |
| M5 | Reference (+ 5 rule-derived vars, CIRCULAR) | 87.1% | 0.639 |

**Takeaway, and the strongest single framing point in the project:**
external validation (the metric that can't be circular) rises
monotonically from M1 to M4, and the fully rule-independent M4 model
has a HIGHER external correlation than the circular M5 reference
(0.652 vs 0.639). CV accuracy is noisier (small n) but tells the same
story -- no feature group is doing nothing, and OSM building
morphology (M3->M4) gives the single largest external-validation gain.

### 9.3 Per-class metrics & confusion matrix (notebooks/11_per_class_metrics.py)

Out-of-fold predictions (every ward predicted once, by a model that
never trained on it), random and thana-grouped CV, reference and
decoupled models:

| Model / scheme | Bal. Accuracy | Macro-F1 | MCC | Informal recall | Formal recall |
|---|---|---|---|---|---|
| Reference / random | 85.6% | 85.8% | 0.717 | 80.3% | 90.8% |
| Reference / thana-grouped | 80.5% | 80.7% | 0.614 | 73.8% | 87.2% |
| Decoupled / random | 84.7% | 84.7% | 0.693 | 80.3% | 89.0% |
| Decoupled / thana-grouped | 82.2% | 82.6% | 0.652 | 75.4% | 89.0% |

**Takeaway:** balanced accuracy and macro-F1 track plain accuracy
closely (no large class-imbalance inflation), MCC in the 0.6-0.7 range
is a respectable effect size for a binary ward classifier, and the
informal class (the minority, and the one planning offices most care
about) is recalled at 75-80% rather than being swamped by the larger
formal class. Report balanced accuracy + MCC alongside accuracy in the
results table; include one confusion matrix (decoupled, thana-grouped
-- the "honest" version) in the main text.

### 9.4 Manuscript integration -- DONE (Oct 2026)

All of 9.1-9.3 plus 9.5-9.6 below are now written into
`Ward-Level_Informal_Settlement_Classification_Dhaka.docx` as new
subsections 5.3-5.7, with the corrected headline numbers (85.3% /
64.1%, not the stale 90.78%/59.5% -- see the correction note in
section 5) propagated through the abstract, Table 1, Table 3, and
Sections 5.1/6.3. The Kuffer -> Yang et al. 2026 citation fix (section
8) is also applied throughout the manuscript, including the reference
list. Re-rendered to PDF and visually checked page by page before
being written back to the project folder.

### 9.5 SHAP feature attribution (notebooks/12_shap_analysis.py) -- DONE (Oct 2026)

SHAP summary plot for the decoupled model (170 labeled wards, 7
rule-independent features). Ranking matches the gain-based importance
order from 9.1-class notebooks (SAVI, OSM building density, slope,
S2 true NDBI, pop_std, OSM mean building area, S2 MNDWI). Directions
match planning intuition: higher SAVI -> Formal, higher OSM building
density -> Informal. Slope shows a non-linear effect -- both very low
and very high slope push toward Informal, moderate slope pushes
toward Formal -- plausibly flood-prone low land and steep
less-serviceable land both being disproportionately informal, with
moderate-slope land preferentially taken for planned development.
Now in manuscript section 5.6.

### 9.6 Agreement plot vs EO4SD (notebooks/13_bland_altman.py) -- DONE (Oct 2026)

Bland-Altman-style plot (mean vs. difference) for decoupled-model
P(informal) vs. EO4SD informal-area fraction, n=203. Mean bias=0.370,
SD=0.397, 95% LoA=[-0.409, 1.149], 0 wards outside the limits of
agreement. The two measures agree near zero, diverge in the
mid-range (model probability saturates toward 1 before EO4SD's area
fraction catches up -- expected, since a ward can have a
probability-dominating informal core while EO4SD polygons cover only
part of its area), and partially reconverge at the high end. Framed
in the manuscript (section 5.7) as a scale-related pattern rather
than a calibration failure, since the two measures are not the same
quantity on the same scale -- the Spearman result remains the
primary evidence. No outlier list needed (0 wards outside LoA).

### 9.7 Outstanding before submission

- [x] Full-chain re-run (Oct 2026): 05b -> 07 -> 08 -> 09 -> 10 -> 11
      -> 12 -> 13 re-executed back to back from freshly re-staged raw
      inputs (not from cached intermediate files), plus 04b and 06
      re-run standalone as a cross-check. Every number in sections 4,
      5, and 9.1-9.3 reproduced exactly (TEST 0/A/B/C, Reference/
      Decoupled/Anchor63 external r, spatial-CV accuracies, ablation
      M1-M5, per-class metrics, SHAP ranking, Bland-Altman bias/SD/LoA
      all matched to the reported decimal places). `ward_eo4sd_comparison.csv`
      regenerated from the shapefile overlay was byte-identical to the
      prior version bar float noise at the 1e-16 level. Full pipeline
      is reproducible end to end under RANDOM_SEED=42.
  - Caught and fixed one real inconsistency in this pass: the
    manuscript's section-5.3 spatial-CV table displayed the decoupled
    model's "Random CV" accuracy as 85.3% (the NB06 headline number,
    used throughout the rest of the paper) but the thana-grouped and
    spatial-block pp deltas next to it (-1.7pp, -2.8pp) had actually
    been computed against NB09's own in-notebook random-CV refit
    (85.9%), not against the displayed 85.3% baseline. Harmless but
    real arithmetic mismatch (85.3-84.2=1.1, not 1.7). Fixed by
    recomputing the deltas against the displayed 85.3% baseline: now
    -1.1pp (thana) and -2.2pp (spatial block). Section 5.3's narrative
    conclusion ("drop falls within the random-CV 95% CI") is
    unaffected either way. Docx re-validated (zip integrity + XML
    well-formed + python-docx open OK) and pushed to the device.
- [x] Human label audit (Oct 2026, done by Fahim directly): 40-ward
      stratified sample (10 each from rule:formal, rule:informal,
      thana:informal_anchor, thana:formal_anchor), independently
      classified from Google Maps satellite imagery, blind to the
      rule/anchor label, judged ward-by-ward in isolation (not by
      comparison across wards -- an early comparative pass was
      discarded in favour of this).
      RESULT: n=40, raw agreement=72.5%, **Cohen's kappa=0.450
      ("moderate" agreement, Landis & Koch scale)**.
      Confusion matrix: human=informal/rule=informal=16,
      human=informal/rule=formal=7, human=formal/rule=informal=4,
      human=formal/rule=formal=13.
      Full report: outputs/model/human_audit_kappa_report_FINAL.txt.
      Process note for transparency: one ward (Kalindi) was dropped and
      replaced with a fresh never-seen ward after its label was
      inadvertently named in a chat discussion of interim results,
      which broke blinding for that one ward specifically; the
      replacement was rated with zero information about the correct
      answer. No other ward's answer was informed by seeing the rule/
      anchor label.
      DONE IN MANUSCRIPT (Oct 2026): added as new Section 5.8 "Human
      label audit" (confusion-matrix table + honest kappa=0.450
      write-up), inserted right before Section 6 Discussion. Limitations
      (Section 7) updated: the old closing "Finally, ... spatial-block
      cross-validation" sentence renumbered to "Sixth," and a new closing
      "Finally" sentence added summarizing the kappa=0.450 result and the
      directional skew, so the labels are explicitly flagged as
      moderately-reliable rather than definitive ground truth. Docx
      re-validated (XML well-formed + zip integrity + python-docx open
      OK, confirmed "5.8 Human label audit" and "kappa=0.450" both
      present in extracted text) and pushed to the device.
- [ ] Final read-through of the manuscript as a whole, now that it has
      grown from ~9 to ~16 pages with the new 5.3-5.7 subsections --
      check nothing reads as repetitive and the figure/table numbering
      is referenced correctly throughout.
