

import pandas as pd
import os
from pathlib import Path



os.environ['QG_JAR']='/PlanRGCN/PlanRGCN/qpe/target/qpe-1.0-SNAPSHOT.jar'
os.environ['QPP_JAR']='/PlanRGCN/qpp/qpp_features/sparql-query2vec/target/sparql-query2vec-0.0.1.jar'
from graph_construction.jar_utils import get_query_graph

