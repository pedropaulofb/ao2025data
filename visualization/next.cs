# 1. Scatter Plot: "Shannon Entropy vs. Total Frequency per Group"
X-Axis: Total Frequency per Group
Y-Axis: Shannon Entropy
Why It’s Useful:
This scatter plot helps identify constructs that are not only frequently used but also evenly distributed across different models.
High Shannon Entropy indicates that the usage of a construct is diverse across different contexts or models, suggesting flexibility or adaptability.
Constructs with high "Total Frequency per Group" and high "Shannon Entropy" are both frequently used and versatile across models, likely representing constructs that are broadly applicable or foundational to different modeling scenarios.
Constructs with low entropy and high frequency could indicate niche constructs that are important in specific cases but not versatile.

# 2. Scatter Plot: "Gini Coefficient vs. Ubiquity Index"
X-Axis: Gini Coefficient
Y-Axis: Ubiquity Index
Why It’s Useful:
This plot helps differentiate constructs based on their equality of usage distribution (Gini Coefficient) versus how widely they are adopted across models (Ubiquity Index).
High Gini Coefficient indicates that a construct's usage is highly unequal across models (used heavily in a few and rarely in others).
Low Gini Coefficient with high Ubiquity Index represents constructs that are both widely adopted and evenly used across models, marking them as "core" constructs.
High Gini Coefficient with low Ubiquity Index may represent specialized constructs used intensely in certain models but not generally.

# 3. Scatter Plot: "Skewness vs. Kurtosis"
X-Axis: Skewness
Y-Axis: Kurtosis
Why It’s Useful:
This plot examines the shape of the distribution of construct usage across models.
Skewness indicates the asymmetry of the distribution (positive skew suggests more frequent low counts, negative skew suggests more frequent high counts).
Kurtosis indicates the "tailedness" or outlier presence in the distribution (high kurtosis suggests many outliers).
Constructs in the low-skew, low-kurtosis quadrant may have a more "normal" or balanced distribution, while those with high skew and kurtosis could represent specialized constructs that show unique or rare usage patterns.

# 4. Scatter Plot: "Simpson's Index vs. Total Frequency"
X-Axis: Total Frequency
Y-Axis: Simpson's Index
Why It’s Useful:
This scatter plot helps identify constructs that are both dominant in terms of occurrences (Total Frequency) and concentrated in their usage (Simpson's Index).
Low Simpson's Index with high Total Frequency suggests constructs that are widely used but not concentrated in a few models — these are potentially "core" constructs.
High Simpson's Index with low frequency suggests constructs that are dominant in a small number of models, possibly pointing to niche applications.

# 5. Scatter Plot: "Mutual Information vs. Jaccard Similarity" (for Construct Pairs)
X-Axis: Jaccard Similarity
Y-Axis: Mutual Information
Why It’s Useful:
This scatter plot provides insights into how pairs of constructs relate to one another in terms of linear co-occurrence (Jaccard Similarity) and non-linear dependencies (Mutual Information).
Construct pairs with high Jaccard Similarity and high Mutual Information might suggest a strong dependency or conceptual relationship.
Pairs with low Jaccard Similarity but high Mutual Information indicate non-linear relationships that aren’t captured by simple co-occurrence, which could be important for refining models or understanding complex interactions.