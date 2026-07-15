# Notes

The model uses only information available before each `pause_start`, so it follows the assignment rules. The main features include pitch, pitch slope, RMS energy, energy trend, voiced ratio, pause position in the conversation, time since the turn started, and simple pause-history features such as the number of previous pauses, previous pause duration, and running average of earlier pause durations.

I trained a `GradientBoostingClassifier` using both the English and Hindi datasets together. I also tried `HistGradientBoostingClassifier`, but it gave lower cross-validation performance, so I continued with the Gradient Boosting model.

At first, the model showed very high scores (around 0.98 AUC), but those results were measured on the same data used for training. After creating a separate held-out evaluation using unseen turns, the AUC became **0.682**, which is a much more realistic estimate of how the model would perform on new data.

The biggest challenge was the small dataset. With only about 200 turns available, the model can still overfit even after regularization. Some pauses are also difficult to classify because their speech patterns are very similar between a short hesitation and the actual end of a turn.

If I had more time, I would try speaker-wise pitch normalization to improve cross-language performance, use probability calibration for better threshold selection, and test the model on a larger dataset. More training data would probably improve the model more than adding many new handcrafted features.