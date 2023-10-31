## Missing Values

-> Near price and far price are missing (NA) for 55% of the data set. I propose making a seperate model for this sub set of the data.

-> Target is missing for 88 observation. I advise dropping these observations.

-> WAP, ask_price, bid_price, matched_size, reference_price, and imbalance_size are missing for 220 observation. They happen all over the dataset, therefore it is advised to drop these rows.

## Modeling Considerations

-> One model per ticker can be made. There are about 28k observations per ticker.

-> far_price has a very big outlier. 