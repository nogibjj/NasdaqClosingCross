## Missing Values

-> Near price and far price variables are missing (NA) for 55% of the data set. Consider making a separate model for this subset of the data.

-> Target is missing for 88 observation. Consider dropping these observations.

-> WAP, ask_price, bid_price, matched_size, reference_price, and imbalance_size are missing for 220 observation. They happen all over the dataset. Consider dropping these rows.


## Modeling Considerations

-> Consider making one model per ticker. There are about 28 thousand observations per ticker.

-> Far_price has a very big outlier. Consider appropriate ways of handling outliers

-> Consider creating a custom loss function more suited to the problem.

-> Consider making new features. Maybe make some based on expert opinion and finding the right distributions for the variables. Use Wolfram alpha to arrive at new distributions visually.

-> Consider removing stock_id or converting it into a categorical variable. It is currently represented as numerical, which does not make sense.

