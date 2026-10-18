# Problem understanding: steel surface defects

Phase 1. Sources are numbered [1] to [12] and listed at the end.

## 1. Context

**How hot-rolled strip is made.** A steel slab is heated in a reheating furnace to roughly 1100 to 1250 °C [1]. High-pressure water sprays (a descaler) then wash off the oxide layer that forms on the hot surface [1]. Rolling mills squeeze the slab thinner in stages: a roughing mill makes a bar about 30 mm thick, and a finishing mill brings it to final thickness [1]. Water sprays cool the strip and a coiler winds it into a coil [1]. "Hot" means the steel is rolled above the temperature at which its grains re-form [2].

**Why surface defects matter.** They make the product look worse and can lower its fatigue resistance, meaning how well it survives repeated loading [3]. They also lead to customer dissatisfaction and economic loss [4]. Descaling in particular is described as a must for good surface quality [1].

**Why manual inspection is weak.** Checking by eye is slow, error-prone and depends on the inspector's experience. Visual fatigue causes missed defects [4]. The strip also moves fast: one article on a hot strip mill quotes strip speeds of up to 15 m/s [5].

## 2. The six NEU defect types

The NEU database has 1,800 grayscale images of 200 x 200 px, 300 for each of 6 classes [6][7]. Where we found no source for a cause, we say so.

| Class | What it looks like | Cause given by our sources |
|---|---|---|
| Crazing | A network of fine surface cracks [6][4] | Not verified. We found no source that explains crazing. |
| Inclusion | Foreign material embedded in the steel surface [6][4] | Non-metallic particles in the steel are linked to internal splits (laminations) and surface slivers [8][9]. Source [8] ties them to casting. |
| Patches | Irregular, blotchy areas [6][4] | Not verified. |
| Pitted surface | Small pits or craters [6][9] | Inclusions that fall out and leave voids, roll surface pitting copied onto the strip, or impact damage [9]. Pits are also tied to scale [10]. |
| Rolled-in scale | Oxide scale pressed into the surface during rolling [6][4] | Weak descaling: low spray pressure or blocked nozzles [9][10], or uneven furnace temperature [8]. |
| Scratches | Linear marks from physical abrasion [6][4] | Debris trapped between roll and strip, or a misaligned roll gap [8]. |

Two classes look alike: rolled-in scale and crazing share visual similarities [4]. Expect them to be confused.

## 3. Link to predictive maintenance

Predictive maintenance means fixing equipment before it fails, using early warning signs. If we count defect types over time, a rising count points at a part of the line:

- more rolled-in scale: check the descaling system (pressure and nozzles) and furnace temperature [8][9][10];
- more scratches: check the rolls, the roll gap alignment and debris on the line [8];
- more pits: check roll surface condition [9];
- more inclusions: the cause is upstream, in steelmaking and casting, so the fix is not in the rolling mill [8].

One defect type can have several causes, and wrong root-cause analysis is common [11]. So a trend is a reason to inspect, not a diagnosis. A mill example exists where inspection images and data are archived so engineers can adjust the furnace process [12]. Our classifier would supply the defect labels for such a trend view. We have not built the trend dashboard, and NEU has no timestamps to test it.

## 4. Our objective

1. Build a CNN (convolutional neural network, a model that learns image patterns) that classifies each image into one of the 6 defect classes.
2. Treat crazing as the "crack" class, since it is the crack-like class in NEU. We also report crack-vs-rest metrics: precision, recall and F1 for crazing against the other five classes. This answers the problem statement directly.

## 5. Success criteria

We set the target numbers later, after seeing the data and a baseline. For now we agree on what to measure:

- **Macro-F1:** F1 averaged over the 6 classes, so no class is ignored.
- **Recall on cracks:** a missed crack (false negative) reaches the customer, while a false alarm only costs a re-check.
- **Fast inference:** time per image must be small, because the strip moves fast.
- **Small model:** it should fit on an edge device (a small computer next to the line).

## 6. Constraints

- NEU images are small 200 x 200 crops, and the dataset page notes large variation in lighting and surface condition [7].
- A line-scan camera builds an image one row of pixels at a time as the strip passes [5], so a real line produces long, high-resolution images. This is our reading of how NEU relates to a real line. A real system would cut such an image into tiles and classify each tile. We will describe this, and may demonstrate it if time allows.
- 300 images per class is a small set, so overfitting (memorising the training images) is the main risk.
- NEU is a benchmark. A real plant would need its own labelled images.

## Sources

1. ispatguru, "Rolling of steel in hot strip mill": furnace temperature, descaling, roughing and finishing stages, coiling. https://www.ispatguru.com/rolling-of-steel-in-hot-strip-mill/
2. Wikipedia, "Rolling (metalworking)": hot rolling above the recrystallization temperature. https://en.wikipedia.org/wiki/Rolling_(metalworking)
3. Classification of Surface Defects in Steel Sheets Using Developed NasNet-Mobile CNN and Few Samples, Revue d'Intelligence Artificielle 38(2): consequences of defects. https://www.iieta.org/journals/ria/paper/10.18280/ria.380232
4. Comparative Analysis of Object Detection Algorithms for Surface Defect Detection, arXiv 2510.21811: class descriptions, manual inspection limits, class similarity.
5. Vision Systems Design, "Linescan cameras inspect steel": line-scan principle and strip speed. https://www.vision-systems.com/factory/article/16736748/linescan-cameras-inspect-steel
6. awesome-industrial-datasets, NEU entry (summarises Song and Yan, 2013): class list. https://github.com/jonathanwvd/awesome-industrial-datasets/blob/master/markdown/neu_surface_defect_dataset.md
7. Same entry as [6]: 1,800 images, 200 x 200, grayscale, lighting and surface variation. IEEE DataPort "steel defect detection" gives the same counts: https://ieee-dataport.org/documents/steel-defect-detection
8. Steel Technology, "Rolling Mill Defects, Process Control, and Quality Improvement": scale pits, scratches, inclusions, cracks and their equipment links. https://www.steel-technology.com/articles/rolling-defects-process-control-steel-quality
9. OXMAINT, "Steel Surface Defect Types and Inspection Guide" (industry guide, not peer reviewed): scale, pitting and inclusion causes. https://oxmaint.com/industries/steel-plant/steel-surface-defect-types-inspection-guide
10. Steel Data Info (VDEh demo database), "Scale pits/scabs": inadequate descaling. https://steeldata.info/macro/demo/data/3509.html
11. ABM Proceedings, "Hot rolling surface quality issues" (abstract only): root-cause analysis is hard. https://abmproceedings.com.br/en/article/download-pdf/hot-rolling-surface-quality-issues-4
12. Vision Systems Design, "Rugged system monitors steel-strip process" (2006): archived inspection data used to adjust the furnace process. https://www.vision-systems.com/boards-software/article/16736631/rugged-system-monitors-steelstrip-process
