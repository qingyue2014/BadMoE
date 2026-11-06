# Expert-probing sets

Each JSON file contains the fixed outcome of randomly sampling 200 examples
from the corresponding clean task data for expert-usage estimation. The
trigger-search script reads the complete file and does not sample it again.

The negative-sentiment and refusal tasks share the same fixed random sample
from Alpaca because both tasks use the same clean instruction source.
