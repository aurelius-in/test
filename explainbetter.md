things to mention in presentation:

1. feature_mentions.csv has three columns one with the list of features and the second column is the numbrer of times each of those features were mentioned in the comments. the third column ranks them with 1 being the most frequently mentioned feature. the features in feature_mentions.csv are sorted with most mentioned first, showing only 36 features mentioned directly or indirectly in the comments for any given provider.  29 of the features are mentioned 2 or more times, 27 of the features are mentioned 3 or more times, 25 are mentioned 4 or more times, 22 are mentioned 5 or more times, 20 are mentioned 6 or more times, 19 are mentioned 7 or more times, 13 are mentioned 9 or more times, 9 are mentioned 12 or more times, 7 are mentioned 36 or more times, 5 are mentioned 43 or more times, 2 are mentioned 183 or more times, and provider type is mentioned 189 times.
this is interesting because we are trying to figure out how to best train a model to accurately predict the riskiest 1% in a set of hundreds of thousand of bh providers and the comments can help us understand the factors that most often go into an agent's findings about risk in cases. Comments offer insight into the decision making process about risk that is otherwise less obvious when considering which of over a hundred features to consider when prioritizing how to train the ml model.
The features mentioned in comments in order of prevelance are as follows:
Feature,Count
Provider Type,189
State,183
% Days >8 Hour/Day,43
Days >8 Hour/Day,43
$ Tot Pd,36
Tot Days Pt Seen,36
Tot Vst Cnt,36
In Ntwk,12
TINs,12
$ X-billing Pd,9
% $ Pd X-billing,9
X-billing Pt Cnt,9
X-billing Vst Cnt,9
$ Non-MD CPT Pd,7
% $ Pd Non-MD CPT,7
Non-MD CPT Pt Cnt,7
Non-MD CPT Vst Cnt,7
Psychother Add On E/M Pt Cnt,6
$ Psychother Add On E/M Pd,5
% $ Pd Psychother Add On E/M,5
Crisis Add On Pt Cnt,5
Psychother Add On E/M Vst Cnt,5
$ Crisis Add On Pd,4
% $ Pd Crisis Add On,4
Crisis Add On Vst Cnt,4
$ High Cost E/M Pd,3
% $ Pd High Cost E/M,3
High Cost E/M Pt Cnt,2
High Cost E/M Vst Cnt,2
$ >8 Hour/Day Days Pd,1
$ Diff State Treat Pd,1
% $ Pd Diff State Treat,1
% Pt Crisis Add On,1
% Pt Psychother Add On E/M,1
Diff State Treat Pt Cnt,1
Diff State Treat Vst Cnt,1


2. features_in_ntwk.csv (arbitrary numbers)
the LLM assigned the scale to have a moderate impact on feature weighting due to the unknown impact of a provider being in network and out-of-network, but since LLM weights are applied to columns in later steps, it makes more sense to give these a more explainable maxiumum, minimum and middle-scale value. Table:
 In Ntwk,Weight,Reason
Y,0.01,Lower risk as in-network providers typically have more oversight and contractual obligations that may reduce opportunities for fraud
N,0.50,Higher risk due to potentially less oversight and regulatory scrutiny for out-of-network providers increasing opportunities for fraud
U,0.99,Highest risk due to the unknown or unclear status of the provider which might correlate with less transparency and higher potential for fraudulent activitieszx

3. comments_score.csv has Comment Scores for 263 providers that were commented on and given a score by the LLM.  The scores take into account the feature_mentions.csv and the raw comments for each provider to calculate a context-based score for each provider.  

3. raw_labeled_providers_all_features.csv contains all the columns for the labeled providers (not only the labelled features but all the other columns as well).  To provide an informed Risk score to each provider that applies to the existing features to which we can consider, risk is added systematically, according to the features, prevelent in the comments.  using all the columns to calculate these values allows for a more informed calculation when preprocessing the labeled data, which provides a more carefully calculated score and an opportunity for more nuanced scoring (perhaps not a 5 point scale, but at least a 3 point scale of high, moderate and low risk which would be preferable over a binary classification, from the business Provider-Practice Management teams usecase stand-point.   

4.risk_level.csv is a file that contains all columns for 2500 providers, including the scaled values for the labeled columns "Reason for Pass", "Comments" "Case Status" and the three action date columns.  The Risk level scaled from 1-to-3 and is calculated using the information above.  
