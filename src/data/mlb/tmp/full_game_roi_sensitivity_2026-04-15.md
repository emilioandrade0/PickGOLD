# Full Game ROI Sensitivity Sweep (2026-04-15)

| scenario | objective | edge | roi_w | acc_w | cov_w | accuracy | pub_roi | priced | roi_split_rate | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| roi_base | roi | 0.0000 | 1.00 | 0.10 | 0.04 | 0.578231 | 0.03698 | 37 | 0.2545 | promote |
| roi_edge_0000_accup | roi | 0.0000 | 1.00 | 0.15 | 0.05 | 0.578231 | 0.03698 | 37 | 0.2545 | promote |
| roi_edge_0000_roiup | roi | 0.0000 | 1.25 | 0.08 | 0.03 | 0.578231 | 0.03698 | 37 | 0.2545 | promote |
| accuracy_cov_ref | accuracy_cov | 0.0000 | 0.00 | 0.00 | 0.00 | 0.578231 | 0.02808 | 21 | 0.0000 | reject |
| roi_edge_0005_basew | roi | 0.0050 | 1.00 | 0.10 | 0.04 | 0.578231 | -0.18653 | 19 | 0.1818 | reject |
| roi_edge_0005_roiup | roi | 0.0050 | 1.25 | 0.08 | 0.03 | 0.578231 | -0.18653 | 19 | 0.1818 | reject |
| roi_edge_0010_basew | roi | 0.0100 | 1.00 | 0.10 | 0.04 | 0.578231 | -0.24064 | 18 | 0.1818 | reject |
| roi_edge_0015_basew | roi | 0.0150 | 1.00 | 0.10 | 0.04 | 0.578231 | -0.38889 | 16 | 0.1636 | reject |
