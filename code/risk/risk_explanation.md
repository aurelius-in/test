#### Handling missing values in expert-labeled data
We explored two distinct approaches for handling missing values in our dataset: mean-based and percentile-based imputation. This comparison is critical for the following reasons:

1. **Data Integrity and Representation**: Missing values can significantly impact the outcomes of our risk assessment model. Choosing the right method to impute these values is crucial to maintain the integrity and representativeness of our data.

2. **Mean-Based Approach**: The mean-based method involves replacing missing values with the average of available data in the same column. This approach is straightforward and often effective, but it assumes that missing values are randomly distributed and similar to the mean of the data. In our context, this could potentially skew the risk scores higher, especially if missing values systematically differ from the mean (e.g., if no action taken typically indicates lower risk).

3. **Percentile-Based Approach**: To address the potential skew from the mean-based method, we explored the percentile-based approach. This involves replacing missing values with a specific percentile (such as the 25th percentile) of the available data. This method was chosen to provide a more conservative estimate that reflects a lower but not minimal risk level, aligning better with our understanding that missing values (no action taken) are more likely to indicate lower risk.

4. **Methodological Rigor**: By comparing these two methods, we aim to demonstrate methodological rigor. It’s essential to test different imputation strategies to understand how they affect our model's output and to choose the one that most accurately reflects the real-world scenarios we are modeling.

5. **Transparency and Justification**: This comparison also serves to enhance transparency in our modeling process. By openly discussing the reasons behind our choice of imputation method, we provide stakeholders with a clear understanding of our decision-making process and reinforce the credibility of our model.

In conclusion, the comparison between mean-based and percentile-based approaches for handling missing values was a crucial step in developing a more accurate and reliable risk assessment model. This exploration helped us align our data treatment with the specific nuances of our dataset and the real-world phenomena it represents.
