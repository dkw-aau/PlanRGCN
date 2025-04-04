# PlanRGCN Task List

## Experiment
**Query Sampling for various experimental results**
For unseen queries, there are few queries in the test set.
We should be able to fix this somewhat in the sampling/data split creation.
- [ ] Given a certain  percentage of desired unseen queries in test, makes the split such that (1) equal amount of fast, med and slow query, (2) contains the unseen queries in the test set. 

**Query Generation for low cardinality in experiment**
We need to add a description of how we want to generate the additional queries.
This requires an automatic approach

Approach (as previously discussed):
- [ ] Define X query templates where subject/object/predicate IRI can be instantiated. Then generate some queries. The definition of templates is the critical aspect of the query generation.
- [ ] Use the trained ML model to check the expected runtime interval classification
- [ ] Based on the expected runtime interval classification, run the run the query once to check whether it follows expectation

**Query Workload Specific Time Intervals/ Generalisation of bins**
Reviewer commented that the time intervals chosen seem arbitrary. To generalize to specific datasets/workloads:
- [ ] Explore different workload dependent strategies [(different types of histograms)](https://en.wikipedia.org/wiki/Histogram)
- [ ] Based on the chosen histogram and a number of time intervals, define workload dependant threshold and use them for experiments.

**Query Characteristics per time interval and operator**

**Additional downstream task**
Reviewer mentioned that additional downstream tasks besides the load balancing task would be ideal because this is where the practicality of the methods can be observed.
- [ ] Option 1 (Admission Control): Load-balancing downstream task is essentially a query scheduling task (we decide the execution order). Another related task could be admission control, similar to [this work (ICDE workshop)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10555064). The benefit of this approach is that it is easier to incorporate in the downstream section in the paper (paper content is a series issue in the paper).
   - [ ] Given a query and a timeout, we decide whether to run the query and not
   - [ ] We then need to define a workload cost function that evaluates the cost of executing a workload, particularly penalizing queries when queries are incorrectly terminated and rewarding correctly terminated queries (maybe not necessary, considering that workload execution time will be reduced). 
   - We need to be smarter about the task definition, as we need the definition to be useful for multiple time intervals beyond two.
- [ ] Option 2 (Query Performance Regression): Turn the problem into a regression problem similar to existing work. This makes it possible to directly compare predictions against baselines. We can do this by taking the existing model and take the last hidden layer representation, and then learn a regression model that takes this hidden representation as input. (NOTE: Baselines can get negative predictions due to the scaling used/ without scaling the baselines are unusable. As per Matteo's earlier suggestion, we might need to map the output/ground truth space to positive values)

**Making experimental results more presentable**
Multiple reviewers for the paper have commented that the confusion matrices are hard to read and have given suggestions on improving this.
- [ ] Make classification metrics (class-wise F1, precision, recall) for each class of each method
- [ ] Make a confusion matrix per method (we will have to be selective about this)
- [ ] Consider other ways to analyse the results that aggregates the results into easily comparable values.

**Online running time for baseline methods**
- [ ] Make results/plots analysing complexity of query representation construction in the inference phase, past the training phase
- [ ] Make a similar analysis for PlanRGCN to compare on different query types, i.e., with many operators vs. few.

**Results on how close the predictions of baselines are to the decision boundaries**
Even though, the majority of the baseline approaches are incorrect. If they are close to the boundaries it means that it is not a significant error.
Therefore, we need some visualization that depicts how close the incorrect predictions for baselines are from the "right" classification.
- [ ] Plot CDF for baselines
- [ ] Figure out a way to compare our approach's error against the baselines that are regression models

**Smaller task**
- [ ] Include total data collection time (3 runs of the queries to get the average)


## Paper

**Query graph definition/construction**
- [ ]  specify property in query graph construction to make this reproducable. The change or clarification should probably happen in relation to the query plan structure and traversal.
- [ ] Reduce the formalism used in query graph construction algo
- [ ] Also we need to reconsider whether all edges make sense. We are currently assuming an implicit order in an unoptimized, query engine-agnostic query plan in that a S-O relation implies that one TP/PP is executed before another. 

**Presentation issue of confusion matrices**
Reviewers have commented that the confusion matrices can be confusion, particularly finding the diagonal considering the three methods (baselines and PlanRGCN).
- [ ] Selectively choose few experimatal results to show with confusion matrices and reorder the table to be more clear (e.g., have one per method as suggested by reviewer)
- [ ] Make table of class-wise F1, Precision and recall for the remaining experiments

**Adaptable to unseen queries**
Specify how the method is adaptable to unseen queries:
- [ ] In the feature construction phase, add a sentence stating that only the statistics of the terms and literals involved in the query is needed for the model
- [ ] Our model learns the query structure through the graph representation and should therefore be able to generalize to unseen query structure (we need to be careful about this formulation, because )

**Emphasize the superiority of PlanRGCN compared to the baselines**

**Formal complexity analysis of algorithm 1 (section 5.1)**
- [ ] Add Big O-analysis of queries 
- [ ] Compare the runtime complexity of our feature representation against the baselines in experiments

**Model architecture choice justificiation**
- [ ] Provide justification for why the binnning feature encoding makes sense. binary encodings are usually more desirable for classification tasks.
- [ ] Apparently the choice of GCN/RGCN is not clear.

**Justification for decomposing PP into multiple operators**
- [ ] Motivate this as a form of densifying the vector representation. We can reuse the triple pattern features that encode the same information for this. Otherwise, we would need a more specialized representation for path queries.
- [ ] We could probably introduce this decomposition as a "natural" step in query parsing

**Add section on QPP in information retrieval in Related Works**
One reviewer commented that it would be a good idea include a section on QPP in information retrieval. It seems a bit outside the domain though.
- [ ] Add a small paragraph about information retrieval (maybe a sentence instead)

**Smaller tasks**
- [ ] Specify what joinable mean in Definition 5.1
- [ ] How the feature extraction from KG is adressed in practice needs to be more clear, where the statistics can relatively straightforward be collected/updated upon insertion. (provide a small description of how this can be achieved)
- [ ] Allegedly, Figure 1 and 2 are not properly referenced/used in text. Double check this and ensure that they are references appropriately for explanations