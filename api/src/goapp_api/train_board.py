"""DEPRECATED — fine-tuned YOLO on the hand-labeled boards dataset.

The board detector is now trained from synthetic pages using
`goapp_api.train_board_synth`. The hand-labeled data in
`training_data/boards_deprecated/` had bbox annotations that clipped board
edges too tightly, causing downstream anchoring problems. It's preserved
for reference only. Don't run this module.
"""

raise SystemExit(
    "train_board.py is deprecated — use "
    "'python -m goapp_api.train_board_synth' instead"
)
