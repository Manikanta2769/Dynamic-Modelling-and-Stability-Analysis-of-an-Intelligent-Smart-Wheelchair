# Graph output behavior

- Main simulation graphs are saved under `output/graphs/`.
- Re-running the same scenario and terrain overwrites the previous graph files.
- Filenames retain scenario, grade, and cross-slope so different test conditions remain distinguishable.
- Controller names are not included in the standard simulation graph filenames, preventing one duplicate set per controller.
- ML fixed-vs-adaptive plots now use stable filenames per controller and overwrite on rerun.
- Old graph files from previous versions are not automatically deleted. You may remove stale duplicates from `output/graphs/` once before your next run.
