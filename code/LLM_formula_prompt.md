I have a csv of over 100k bh providers, each with over 100 columns describing the provider geo-demographics and billiing activity and practice behavior such as Provider Type, State, Region, In Ntwk, $ >3 Diag Eval Pd, $ >8 Hour/Day Days Pd, $ Concurr Treat Pd, $ Crisis Add On Pd, $ Crisis Overall Pd, etc. 
Most of these providers have had no manual determination of risk made by expert insurance agents (no actions taken).
For those providers who have had some actions taken there maybe a value in up to one or more additional 'actions' columns inclucing:
Comments, Most Recent Case Status, Reasons for pass, Most Recent Case Open Dt, Most Recent Case Close Dt, Most Recent Data Mining Activity Update Dt.

A weight is given to each value using the LLM to determine its importance for determining risk. Most of the providers on the list of labeled providers are missing one or more of these values.  
Any provider with one or more value in an actions column is considered labelled data, but a determination must be made how to assing a risk score to each provider based on the values in each of these columns.
Only a small minority of providers have a value in one or more of the six actions columns (~2500 out of 111k are 'labeled'), therfore a formula is needed to determine a risk score based on the six values.  
The values have been preprocessed by an LLM to determine a value for each, so the Comments have been converted to a Comments Score, the Most Recent Case Status has been given a Score, the Reason for Pass has been given a score, etc.
Create a forumula to assign a risk score to each provider, giving the most importance to the Most Recent Case Status, Reason for Pass score, Comments score, amd the Most Recent Closed Status Score.
Also account for the fact that for many of the labeled providers, one or more of these values will be missing and a Risk score should be assigned even when all values are not available.
