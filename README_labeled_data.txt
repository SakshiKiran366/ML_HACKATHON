ML_HACKATHON labeled dataset

Files
- candidates_train_labeled.parquet: candidate road arcs for training trips, with binary labels.
- candidates_student_test_labeled.parquet: candidate road arcs for student_test trips, with binary labels.
- candidate_diagnostics.csv: per-trip label and candidate-generation diagnostics.

Label meaning
Each row is one (GPS observation, candidate road arc) pair. label=1 marks the arc selected by monotone alignment of the GPS observations to that trip's supplied reference route; label=0 marks another candidate. Features are built from GPS track and road network data. Rows for observations farther than 60 metres from the aligned reference arc are omitted as unreliable, per the repository pipeline.

The student_test files have labels because the repository includes .route files for those trips. Check the event rules before using student_test labels for training or model selection; the repository README says student_test is for development comparison only.

Source: https://github.com/SakshiKiran366/ML_HACKATHON
Generated from the repository's existing tables/ and results/ artifacts.
