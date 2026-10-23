# Notebook Demonstration Media

These files are derived from the annotated exports produced by
[`notebooks/shopper_behavior.ipynb`](../../notebooks/shopper_behavior.ipynb).
They are demonstration artifacts, independent of the backend's test fixtures and
application database.

| Scenario | Processed export | Frame count / duration | Source recorded in the notebook |
|---|---|---|---|
| Retail store overview | [MP4](retail-store-processed.mp4) | 950 frames / 31.7 s at 30 FPS | [ACTi retail-store sample](https://www.youtube.com/watch?v=-1bRhYjw1qE) |
| Aisle close-up | [MP4](aisle-closeup-processed.mp4) | 6,374 frames / 212.5 s at 30 FPS | [iProx retail CCTV sample](https://www.youtube.com/watch?v=B9xQM8GEiYk) |

The exports preserve pose/track overlays, product/presence polygons, projected head
direction, and the notebook's original heuristic labels. Those labels are not
verified detections or calibrated emotional states. In particular, a legacy
`PRODUCT PICKUP` annotation does not prove product removal, contact or purchase.
The stricter backend action contract requires separately supported object evidence.

`retail-store-preview.gif` shows seconds 6–12 of its processed export.
`aisle-closeup-preview.gif` shows seconds 25–31. Previews use 640-pixel width,
8 FPS and a palette optimized for the excerpt. They contain no additional inference.
Select the MP4 for full resolution and normal frame timing. Exports contain video
only and use browser-compatible H.264/YUV420p with MP4 fast-start metadata.

## Saved Notebook Results

The PNG figures are extracted directly from the completed notebook's saved
outputs, without recomputing or changing the plotted data:

| Scenario | Spatial results | Biomechanical results |
|---|---|---|
| Retail store overview | [Action-grouped heatmaps](retail-store-heatmaps.png) | [Tracked-shopper time series](retail-store-kinematics.png) |
| Aisle close-up | [Action-grouped heatmaps](aisle-closeup-heatmaps.png) | [Tracked-shopper time series](aisle-closeup-kinematics.png) |

Heatmaps accumulate estimated head-direction projections weighted by the legacy
engagement score. Each panel is normalized independently; intensity does not
provide an absolute comparison between actions or cameras. Counts represent
track-frame observations rather than unique visitors. Time series select the
track with the most observations and show estimated joint angles and distances
normalized by projected shoulder width. Thresholds and action labels are POC
heuristics, without ground-truth validation.

Only these reviewed MP4 paths are exempted from the repository's generated-video
ignore rule. Original downloads, raw encodings, model weights, backups and local
execution records stay outside the publication set.

Source footage and visible third-party marks retain their original rights. The
project's MIT license covers project code and does not relicense the source footage.
Source links identify the inputs recorded by the notebook; they are not an assertion
of a particular upstream redistribution license.
