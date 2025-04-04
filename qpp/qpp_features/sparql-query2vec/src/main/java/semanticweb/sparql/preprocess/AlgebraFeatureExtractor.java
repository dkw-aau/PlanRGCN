package semanticweb.sparql.preprocess;

import org.apache.commons.collections.map.DefaultedMap;
import org.apache.jena.query.*;
import org.apache.jena.sparql.algebra.Algebra;
import org.apache.jena.sparql.algebra.Op;
import org.apache.jena.sparql.algebra.OpWalker;
import org.apache.jena.sparql.algebra.op.*;
import semanticweb.sparql.config.ProjectConfiguration;
import java.util.*;
import java.util.Map.Entry;


public class AlgebraFeatureExtractor {


    private Map<String, Integer> featureIndex = new HashMap<>();
    private Map<Op, Boolean> visited = null;
	private boolean debug = false;
	private int treeHeight = 0;


    public AlgebraFeatureExtractor(String[] features) {
        //Creating order of features.
        for (int i = 0; i < features.length; i++) {
            featureIndex.put(features[i], i);
        }
    }

    public void setDebug(boolean debug) {
        this.debug = debug;
    }

    public double[] extractFeatures(String queryStr) {
        Query query = QueryFactory.create(queryStr, Syntax.syntaxARQ);
        // Generate algebra
        Op op = Algebra.compile(query);
//        Plan a = QueryExecutionFactory.createPlan(query, DatasetGraphFactory.createTxnMem(), new BindingHashMap());
//        Op op = a.getOp();
        if (debug) {
//            System.out.println("Algebra tree:\n " + a );
        }
        //System.out.println("------------------------") ;
        double[] features = getFeatures(op);


        treeHeight = 0;
        visited = new DefaultedMap(false);
        //System.out.println("Visited size: "+visited.size());
        walkAlgebraTreeRecursive(op, 1);
        features[featureIndex.get("treesize")] += treeHeight;
        //System.out.println("Height:"+treeHeight);


        return features;
    }

    private double[] getFeatures(Op op) {
//        final double[] features = new double[featureIndex.size()];
        
        OpWalker ow = new OpWalker();
        OperatorsQueryVisitor visitor = new OperatorsQueryVisitor(featureIndex);
        ow.walk(op, visitor);
        double[] features = visitor.getFeatures();
       /* OpWalker.walk(op, new OpVisitorBase() {

            public void visit(OpTriple opTriple) {
                //System.out.print("triple ");
                //System.out.println(opTriple);
                features[featureIndex.get("triple")] += 1.0;
            }

            public void visit(OpBGP opBGP) {
                //System.out.println("bgp");

                features[featureIndex.get("bgp")]++;
                features[featureIndex.get("triple")] += opBGP.getPattern().size();

                //System.out.println(opBGP.getPattern());

                //System.out.println(opBGP.getPattern().size());
            }

            public void visit(OpJoin opJoin) {
                //System.out.println("join ");
                features[featureIndex.get("join")] += 1.0;


            }

            public void visit(OpLeftJoin opleftJoin) {
                //System.out.println("leftjoin ");
                features[featureIndex.get("leftjoin")] += 1.0;

            }

            public void visit(OpUnion opUnion) {
                //System.out.println("union ");
                features[featureIndex.get("union")] += 1.0;

            }

            public void visit(OpFilter opFilter) {
                //System.out.println("filter ");
                features[featureIndex.get("filter")] += 1.0;
            }

            public void visit(OpGraph opGraph) {
                //System.out.println("graph ");
                features[featureIndex.get("graph")] += 1.0;
            }

            public void visit(OpExtend opExtend) {
                //System.out.println("extend ");
                features[featureIndex.get("extend")] += 1.0;
            }

            public void visit(OpMinus opMinus) {
                //System.out.println("minus ");
                features[featureIndex.get("minus")] += 1.0;
            }

            public void visit(OpPath opPath) {
                //System.out.println("path ");
                //System.out.println(opPath);
                //System.out.println(opPath.getName());
                Path path = opPath.getTriplePath().getPath();

                if (path instanceof P_ZeroOrMore1) {
                    //System.out.println("path*");
                    features[featureIndex.get("path*")] += 1.0;
                } else if (path instanceof P_ZeroOrMoreN) {
                    //System.out.println("pathN*");
                    features[featureIndex.get("pathN*")] += 1.0;
                } else if (path instanceof P_OneOrMore1) {
                    //System.out.println("path+");
                    features[featureIndex.get("path+")] += 1.0;
                } else if (path instanceof P_OneOrMoreN) {
                    //System.out.println("pathN+");
                    features[featureIndex.get("pathN+")] += 1.0;
                } else if (path instanceof P_ZeroOrOne) {
                    //System.out.println("path?");
                    features[featureIndex.get("path?")] += 1.0;
                } else if (path instanceof P_Multi) {
                    //System.out.println("multi");
                    features[featureIndex.get("multi")] += 1.0;
                } else {
                    //System.out.println("notoneof");
                    features[featureIndex.get("notoneof")] += 1.0;
                }
            }

            public void visit(OpList opList) {
                //System.out.println("tolist");
                features[featureIndex.get("tolist")] += 1.0;
            }

            public void visit(OpOrder opOrder) {
                //System.out.println("order");
                features[featureIndex.get("order")] += 1.0;
            }


            public void visit(OpProject opProject) {

                //System.out.print("project ");
                //List<Var> vars = opProject.getVars();
                //for (Var var:vars) {
                //	System.out.print(" "+var);
                //}
                //System.out.println();

                features[featureIndex.get("project")] += 1.0;

            }


            public void visit(OpDistinct opDistinct) {

                //System.out.println("distinct ");
                features[featureIndex.get("distinct")] += 1.0;

            }

            public void visit(OpReduced opReduce) {

                //System.out.println("reduced ");
                features[featureIndex.get("reduced")] += 1.0;

            }

            //multi is in OpPath


            public void visit(OpTopN opTop) {

                //System.out.print("top ");
                double limit = opTop.getLimit() > 0 ? (double) opTop.getLimit() : 1.0;
                //System.out.println(limit);
                features[featureIndex.get("top")] += limit;

            }

            public void visit(OpGroup opGroup) {

                System.out.println("group ");
                features[featureIndex.get("group")] += 1.0;

            }

            public void visit(OpAssign opAssign) {

                System.out.println("assign ");
                features[featureIndex.get("assign")] += 1.0;

            }

            public void visit(OpSequence opSequence) {

                System.out.println("sequence ");
                features[featureIndex.get("sequence")] += 1.0;

            }

            public void visit(OpConditional opConditional) {
                System.out.println("conditional");
            }


            public void visit(OpSlice opSlice) {

                //System.out.println(opSlice.getSubOp());
                long start = opSlice.getStart() < 0 ? 0 : opSlice.getStart();
                long end = opSlice.getLength();
                double total = (double) start + (double) end;
                //System.out.println("slice "+start+" "+end);
                //System.out.println("Total:"+total);
                features[featureIndex.get("slice")] += total;

            }


        });*/
        return features;

    }

    private void walkAlgebraTreeRecursive(Op op, final int level) {
        visited.put(op, true);
        //System.out.println("Level:"+level);
        //System.out.println(op);


        if (treeHeight < level) {
            treeHeight = level;
        }

        if (op instanceof OpTopN) {
            OpTopN arg0 = (OpTopN) op;
            if (!visited.get(arg0.getSubOp())) walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);
        }

        if (op instanceof OpGroup) {
            OpGroup arg0 = (OpGroup) op;
            if (!visited.get(arg0.getSubOp())) walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);

        }

        if (op instanceof OpSlice) {
            OpSlice arg0 = (OpSlice) op;

            if (!visited.get(arg0.getSubOp())) walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);
        }

        if (op instanceof OpDistinct) {
            OpDistinct arg0 = (OpDistinct) op;

            if (!visited.get(arg0.getSubOp())) walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);

        }

        if (op instanceof OpReduced) {
            OpReduced arg0 = (OpReduced) op;

            if (!visited.get(arg0.getSubOp())) walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);
        }

        if (op instanceof OpProject) {
            OpProject arg0 = (OpProject) op;
            //System.out.println("OpProject sub op:"+arg0.getSubOp());
            //System.out.println("Visited size (OpProject): "+visited.size());
            if (!visited.get(arg0.getSubOp())) {
                walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);
            }
        }

        if (op instanceof OpOrder) {
            OpOrder arg0 = (OpOrder) op;

            if (!visited.get(arg0.getSubOp())) walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);

        }

        if (op instanceof OpList) {
            OpList arg0 = (OpList) op;

            if (!visited.get(arg0.getSubOp())) walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);
        }

        if (op instanceof OpDisjunction) {
            OpDisjunction arg0 = (OpDisjunction) op;

            List<Op> ops = arg0.getElements();
            for (Op o : ops) {
                if (!visited.get(o))
                    walkAlgebraTreeRecursive(o, level + 1);
            }

        }

        if (op instanceof OpSequence) {
            OpSequence arg0 = (OpSequence) op;
            List<Op> ops = arg0.getElements();
            for (Op o : ops) {
                if (!visited.get(o))
                    walkAlgebraTreeRecursive(o, level + 1);
            }

        }

        if (op instanceof OpConditional) {
            OpConditional arg0 = (OpConditional) op;
            if (!visited.get(arg0.getLeft())) walkAlgebraTreeRecursive(arg0.getLeft(), level + 1);
            if (!visited.get(arg0.getRight())) walkAlgebraTreeRecursive(arg0.getRight(), level + 1);


        }

        if (op instanceof OpMinus) {
            OpMinus arg0 = (OpMinus) op;
            if (!visited.get(arg0.getLeft())) walkAlgebraTreeRecursive(arg0.getLeft(), level + 1);
            if (!visited.get(arg0.getRight())) walkAlgebraTreeRecursive(arg0.getRight(), level + 1);

        }

        if (op instanceof OpDiff) {
            OpDiff arg0 = (OpDiff) op;
            if (!visited.get(arg0.getLeft())) walkAlgebraTreeRecursive(arg0.getLeft(), level + 1);
            if (visited.get(arg0.getRight()) == false) walkAlgebraTreeRecursive(arg0.getRight(), level + 1);

        }

        if (op instanceof OpUnion) {
            OpUnion arg0 = (OpUnion) op;
            if (!visited.get(arg0.getLeft())) walkAlgebraTreeRecursive(arg0.getLeft(), level + 1);
            if (!visited.get(arg0.getRight())) walkAlgebraTreeRecursive(arg0.getRight(), level + 1);

        }

        if (op instanceof OpLeftJoin) {
            OpLeftJoin arg0 = (OpLeftJoin) op;
            if (!visited.get(arg0.getLeft())) walkAlgebraTreeRecursive(arg0.getLeft(), level + 1);
            if (!visited.get(arg0.getRight())) walkAlgebraTreeRecursive(arg0.getRight(), level + 1);

        }

        if (op instanceof OpJoin) {
            OpJoin arg0 = (OpJoin) op;
            if (!visited.get(arg0.getLeft())) walkAlgebraTreeRecursive(arg0.getLeft(), level + 1);
            if (!visited.get(arg0.getRight())) walkAlgebraTreeRecursive(arg0.getRight(), level + 1);

        }

        if (op instanceof OpExtend) {
            OpExtend arg0 = (OpExtend) op;
            if (!visited.get(arg0.getSubOp())) walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);

        }

        if (op instanceof OpAssign) {
            OpAssign arg0 = (OpAssign) op;
            if (!visited.get(arg0.getSubOp())) walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);

        }

        if (op instanceof OpLabel) {
            OpLabel arg0 = (OpLabel) op;
            if (!visited.get(arg0.getSubOp())) walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);

        }

        if (op instanceof OpDatasetNames) {

        }

        if (op instanceof OpService) {
            OpService arg0 = (OpService) op;
            if (!visited.get(arg0.getSubOp())) walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);

        }

        if (op instanceof OpGraph) {
            OpGraph arg0 = (OpGraph) op;
            if (!visited.get(arg0.getSubOp())) walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);

        }

        if (op instanceof OpFilter) {
            OpFilter arg0 = (OpFilter) op;
            if (!visited.get(arg0.getSubOp())) walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);


        }

        if (op instanceof OpPropFunc) {
            OpPropFunc arg0 = (OpPropFunc) op;
            if (!visited.get(arg0.getSubOp())) walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);

        }

        if (op instanceof OpProcedure) {
            OpProcedure arg0 = (OpProcedure) op;
            if (!visited.get(arg0.getSubOp())) walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);
        }
		
		if(op instanceof OpNull) {

		}

		if(op instanceof OpTable) {

		}

		if(op instanceof OpPath) {

		}

		if(op instanceof OpQuad) {

		}

		if(op instanceof OpTriple) {


		}

		if(op instanceof OpQuadBlock) {

		}

		if(op instanceof OpQuadPattern) {

		}

		if(op instanceof OpBGP) {

		}
    }

/*    private void walkAlgebraTreeOld(Op op, final int level) {

        visited.put(op, true);
        System.out.println("Level:" + level);
        System.out.println(op);


        if (treeHeight < level) {
            treeHeight = level;
        }


        OpWalker.walk(op, new OpVisitorBase() {


            public void visit(OpTopN arg0) {
                walkAlgebraTreeOld(arg0.getSubOp(), level + 1);
            }

            public void visit(OpGroup arg0) {

                walkAlgebraTreeOld(arg0.getSubOp(), level + 1);

            }

            public void visit(OpSlice arg0) {

                walkAlgebraTreeOld(arg0.getSubOp(), level + 1);
            }

            public void visit(OpDistinct arg0) {

                walkAlgebraTreeOld(arg0.getSubOp(), level + 1);

            }

            public void visit(OpReduced arg0) {

                walkAlgebraTreeOld(arg0.getSubOp(), level + 1);
            }

            public void visit(OpProject arg0) {
                System.out.println("OpProject sub op:" + arg0.getSubOp());
                System.out.println("Visited size (OpProject): " + visited.size());

                walkAlgebraTreeOld(arg0.getSubOp(), level + 1);

            }

            public void visit(OpOrder arg0) {

                walkAlgebraTreeOld(arg0.getSubOp(), level + 1);

            }

            public void visit(OpList arg0) {

                walkAlgebraTreeOld(arg0.getSubOp(), level + 1);
            }

            public void visit(OpExt arg0) {

            }

            public void visit(OpDisjunction arg0) {

                List<Op> ops = arg0.getElements();
                for (Op o : ops) {

                    walkAlgebraTreeOld(o, level + 1);
                }
            }

            public void visit(OpSequence arg0) {
                List<Op> ops = arg0.getElements();
                for (Op o : ops) {

                    walkAlgebraTreeOld(o, level + 1);
                }

            }

            public void visit(OpConditional arg0) {
                walkAlgebraTreeOld(arg0.getLeft(), level + 1);
                walkAlgebraTreeOld(arg0.getRight(), level + 1);
            }

            public void visit(OpMinus arg0) {
                walkAlgebraTreeOld(arg0.getLeft(), level + 1);
                walkAlgebraTreeOld(arg0.getRight(), level + 1);
            }

            public void visit(OpDiff arg0) {
                walkAlgebraTreeOld(arg0.getLeft(), level + 1);
                walkAlgebraTreeOld(arg0.getRight(), level + 1);
            }

            public void visit(OpUnion arg0) {
                walkAlgebraTreeOld(arg0.getLeft(), level + 1);
                walkAlgebraTreeOld(arg0.getRight(), level + 1);
            }

            public void visit(OpLeftJoin arg0) {
                walkAlgebraTreeOld(arg0.getLeft(), level + 1);
                walkAlgebraTreeOld(arg0.getRight(), level + 1);
            }

            public void visit(OpJoin arg0) {
                walkAlgebraTreeOld(arg0.getLeft(), level + 1);
                walkAlgebraTreeOld(arg0.getRight(), level + 1);
            }

            public void visit(OpExtend arg0) {
                walkAlgebraTreeOld(arg0.getSubOp(), level + 1);
            }

            public void visit(OpAssign arg0) {
                walkAlgebraTreeOld(arg0.getSubOp(), level + 1);
            }

            public void visit(OpLabel arg0) {
                walkAlgebraTreeOld(arg0.getSubOp(), level + 1);
            }

            public void visit(OpDatasetNames arg0) {

            }

            public void visit(OpService arg0) {
                walkAlgebraTreeOld(arg0.getSubOp(), level + 1);
            }

            public void visit(OpGraph arg0) {
                walkAlgebraTreeOld(arg0.getSubOp(), level + 1);
            }

            public void visit(OpFilter arg0) {
                walkAlgebraTreeOld(arg0.getSubOp(), level + 1);
            }

            public void visit(OpPropFunc arg0) {
                walkAlgebraTreeOld(arg0.getSubOp(), level + 1);
            }

            public void visit(OpProcedure arg0) {
                walkAlgebraTreeOld(arg0.getSubOp(), level + 1);
            }

            public void visit(OpNull arg0) {

            }

            public void visit(OpTable arg0) {

            }

            public void visit(OpPath arg0) {

            }

            public void visit(OpQuad arg0) {

            }

            public void visit(OpTriple arg0) {

            }

            public void visit(OpQuadBlock arg0) {

            }

            public void visit(OpQuadPattern arg0) {

            }

            public void visit(OpBGP arg0) {
                System.out.println(arg0.toString());
            }
        });
    }*/

    public static void main(String[] args) {
        String[] headers = (String[]) ProjectConfiguration.getAlgebraFeatureHeader();

        AlgebraFeatureExtractor fe = new AlgebraFeatureExtractor(headers);

        fe.setDebug(true);
        String queryStr = "PREFIX foaf: <http://xmlns.com/foaf/0.1/> SELECT DISTINCT ?x ?name  WHERE { ?x foaf:mbox <mailto:person@server.com> . ?x foaf:name ?name . }";
        System.out.println(queryStr);
        double[] features = fe.extractFeatures(queryStr);

        for (double f : features) {
            System.out.print(" " + f);
        }

        int count = 0;
        System.out.println();
        List<Entry<String, Integer>> ll = new ArrayList<>(fe.featureIndex.entrySet());
        //fe.featureIndex.entrySet();

        ll.sort(new Comparator<Entry<String, Integer>>() {
			@Override
			public int compare(Entry<String, Integer> o1,
							   Entry<String, Integer> o2) {
				// TODO Auto-generated method stub
				return o1.getValue() - o2.getValue();
			}
		});

        for (Entry<String, Integer> e : ll) {
            String header = e.getKey();
            if (count != 0) {
                System.out.print(",");
            }
            System.out.print(header);
            count++;
        }
    }
}
