package semanticweb.sparql.preprocess;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.commons.collections.map.DefaultedMap;

import org.apache.jena.query.ARQ;
import semanticweb.sparql.utils.DBPediaUtils;

/*import com.hp.hpl.jena.query.Query;
import com.hp.hpl.jena.query.QueryFactory;
import com.hp.hpl.jena.sparql.algebra.Algebra;
import com.hp.hpl.jena.sparql.algebra.Op;
import com.hp.hpl.jena.sparql.algebra.OpVisitor;
import com.hp.hpl.jena.sparql.algebra.OpVisitorBase;
import com.hp.hpl.jena.sparql.algebra.OpWalker;
import com.hp.hpl.jena.sparql.algebra.op.OpAssign;
import com.hp.hpl.jena.sparql.algebra.op.OpBGP;
import com.hp.hpl.jena.sparql.algebra.op.OpBase;
import com.hp.hpl.jena.sparql.algebra.op.OpConditional;
import com.hp.hpl.jena.sparql.algebra.op.OpDatasetNames;
import com.hp.hpl.jena.sparql.algebra.op.OpDiff;
import com.hp.hpl.jena.sparql.algebra.op.OpDisjunction;
import com.hp.hpl.jena.sparql.algebra.op.OpDistinct;
import com.hp.hpl.jena.sparql.algebra.op.OpExt;
import com.hp.hpl.jena.sparql.algebra.op.OpExtend;
import com.hp.hpl.jena.sparql.algebra.op.OpFilter;
import com.hp.hpl.jena.sparql.algebra.op.OpGraph;
import com.hp.hpl.jena.sparql.algebra.op.OpGroup;
import com.hp.hpl.jena.sparql.algebra.op.OpJoin;
import com.hp.hpl.jena.sparql.algebra.op.OpLabel;
import com.hp.hpl.jena.sparql.algebra.op.OpLeftJoin;
import com.hp.hpl.jena.sparql.algebra.op.OpList;
import com.hp.hpl.jena.sparql.algebra.op.OpMinus;
import com.hp.hpl.jena.sparql.algebra.op.OpNull;
import com.hp.hpl.jena.sparql.algebra.op.OpOrder;
import com.hp.hpl.jena.sparql.algebra.op.OpPath;
import com.hp.hpl.jena.sparql.algebra.op.OpProcedure;
import com.hp.hpl.jena.sparql.algebra.op.OpProject;
import com.hp.hpl.jena.sparql.algebra.op.OpPropFunc;
import com.hp.hpl.jena.sparql.algebra.op.OpQuad;
import com.hp.hpl.jena.sparql.algebra.op.OpQuadBlock;
import com.hp.hpl.jena.sparql.algebra.op.OpQuadPattern;
import com.hp.hpl.jena.sparql.algebra.op.OpReduced;
import com.hp.hpl.jena.sparql.algebra.op.OpSequence;
import com.hp.hpl.jena.sparql.algebra.op.OpService;
import com.hp.hpl.jena.sparql.algebra.op.OpSlice;
import com.hp.hpl.jena.sparql.algebra.op.OpTable;
import com.hp.hpl.jena.sparql.algebra.op.OpTopN;
import com.hp.hpl.jena.sparql.algebra.op.OpTriple;
import com.hp.hpl.jena.sparql.algebra.op.OpUnion;
import com.hp.hpl.jena.sparql.core.Var;
import com.hp.hpl.jena.sparql.path.P_Multi;
import com.hp.hpl.jena.sparql.path.P_OneOrMore1;
import com.hp.hpl.jena.sparql.path.P_OneOrMoreN;
import com.hp.hpl.jena.sparql.path.P_ZeroOrMore1;
import com.hp.hpl.jena.sparql.path.P_ZeroOrMoreN;
import com.hp.hpl.jena.sparql.path.P_ZeroOrOne;
import com.hp.hpl.jena.sparql.path.Path;*/
import org.apache.jena.query.Query;
import org.apache.jena.query.QueryFactory;
import org.apache.jena.sparql.algebra.Algebra;
import org.apache.jena.sparql.algebra.Op;
import org.apache.jena.sparql.algebra.OpVisitor;
import org.apache.jena.sparql.algebra.OpVisitorBase;
import org.apache.jena.sparql.algebra.OpWalker;
import org.apache.jena.sparql.algebra.op.*;
import org.apache.jena.sparql.core.Var;
import org.apache.jena.sparql.path.*;


public class AlgebraFeatureExtractor {


	public Map<String, Integer> getFeatureIndex() {
		return featureIndex;
	}

	Map<String, Integer> featureIndex = new HashMap<String, Integer>();
	Map<Op,Boolean> visited = null;
	boolean debug = false;
	
	public AlgebraFeatureExtractor() {
	
		featureIndex.put("triple", 0);
		featureIndex.put("bgp", 1);
		featureIndex.put("join", 2);
		featureIndex.put("leftjoin", 3);
		featureIndex.put("union", 4);
		featureIndex.put("filter", 5);
		featureIndex.put("graph", 6);
		featureIndex.put("extend", 7);
		featureIndex.put("minus", 8);
		featureIndex.put("path*", 9);
		featureIndex.put("pathN*", 10);
		featureIndex.put("path+", 11);
		featureIndex.put("pathN+", 12);
		featureIndex.put("notoneof", 13);
		featureIndex.put("tolist", 14);
		featureIndex.put("order", 15);
		featureIndex.put("project", 16);
		featureIndex.put("distinct", 17);
		featureIndex.put("reduced", 18);
		featureIndex.put("multi", 19);
		featureIndex.put("top", 20);
		featureIndex.put("group", 21);
		featureIndex.put("assign", 22);
		featureIndex.put("sequence", 23);
		featureIndex.put("slice", 24);
		featureIndex.put("treesize", 25);

		ARQ.init();
	}
	public void setDebug(boolean debug) {
		this.debug = debug;
	}
	int treeHeight = 0;
	public double[] extractFeatures(String queryStr){
		Query query = QueryFactory.create(queryStr) ;
		
        // Generate algebra
        Op op = Algebra.compile(query) ;
        //op = Algebra.optimize(op) ;
        //System.out.println(op.getName());
        
        
        if(debug) {
        	System.out.println("Algebra tree:\n "+op) ;
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
		final double[] features = new double[featureIndex.size()];
		
		OpWalker.walk(op, new OpVisitorBase() {
			
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
				
				if (path instanceof  P_ZeroOrMore1) {
					//System.out.println("path*");
					features[featureIndex.get("path*")] += 1.0;
				} else if (path instanceof  P_ZeroOrMoreN) {
					//System.out.println("pathN*");
					features[featureIndex.get("pathN*")] += 1.0;
				}else if (path instanceof  P_OneOrMore1) {
					//System.out.println("path+");
					features[featureIndex.get("path+")] += 1.0;
				} else if (path instanceof  P_OneOrMoreN) {
					//System.out.println("pathN+");
					features[featureIndex.get("pathN+")] += 1.0;
				} else if (path instanceof  P_ZeroOrOne) {
					//System.out.println("path?");
					features[featureIndex.get("path?")] += 1.0;
				} else if (path instanceof  P_Multi) {
					//System.out.println("multi");
					features[featureIndex.get("multi")] += 1.0;
				}  else {
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
			

			public void visit(OpProject opProject)  {
				
				//System.out.print("project ");
				//List<Var> vars = opProject.getVars();
				//for (Var var:vars) {
				//	System.out.print(" "+var);
				//}
				//System.out.println();
				
				features[featureIndex.get("project")] += 1.0;
				
			}
			

			public void visit(OpDistinct opDistinct)  {
				
				//System.out.println("distinct ");
				features[featureIndex.get("distinct")] += 1.0;
				
			}
			public void visit(OpReduced opReduce)  {
				
				//System.out.println("reduced ");
				features[featureIndex.get("reduced")] += 1.0;
				
			}

			//multi is in OpPath
			
			
			public void visit(OpTopN opTop)  {
				
				//System.out.print("top ");
				double limit = opTop.getLimit()>0?(double)opTop.getLimit():1.0;
				//System.out.println(limit);
				features[featureIndex.get("top")] += limit;
				
			}		
			
			public void visit(OpGroup opGroup)  {
				
				//System.out.println("group ");
				features[featureIndex.get("group")] += 1.0;
				
			}		

			public void visit(OpAssign opAssign)  {
				
				//System.out.println("assign ");
				features[featureIndex.get("assign")] += 1.0;
				
			}		
			
			public void visit(OpSequence opSequence)  {
				
				//System.out.println("sequence ");
				features[featureIndex.get("sequence")] += 1.0;
				
			}		
			
			public void visit(OpConditional opConditional) {
				//System.out.println("conditional");
			}
			

			
			public void visit(OpSlice opSlice) {
				
				//System.out.println(opSlice.getSubOp());
				long start = opSlice.getStart()<0?0:opSlice.getStart();
				long end = opSlice.getLength();
				double total = (double)start+(double)end;
				//System.out.println("slice "+start+" "+end);
				//System.out.println("Total:"+total);
				features[featureIndex.get("slice")] += total;
				
			}
			

		});
		return features;
		
	}
	
	private void walkAlgebraTreeRecursive(Op op, final int level) {
		visited.put(op, true);
		//System.out.println("Level:"+level);
		//System.out.println(op);
		
		
		if(treeHeight<level) {
			treeHeight = level;
		}
		
		if(op instanceof OpTopN) {
			OpTopN arg0 = (OpTopN) op;
			if(visited.get(arg0.getSubOp())==false) walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);
		}
		
		if(op instanceof OpGroup ) {
			OpGroup arg0 = (OpGroup) op;
			if(visited.get(arg0.getSubOp())==false) walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);
			
		}
		
		if(op instanceof OpSlice ) {
			OpSlice arg0 = (OpSlice) op;
			
			if(visited.get(arg0.getSubOp())==false) walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);
		}
		
		if(op instanceof OpDistinct) {
			OpDistinct arg0 = (OpDistinct) op;
			
			if(visited.get(arg0.getSubOp())==false) walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);
			
		}
		
		if(op instanceof OpReduced ) {
			OpReduced arg0 = (OpReduced ) op;
			
			if(visited.get(arg0.getSubOp())==false) walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);
		}
		
		if(op instanceof OpProject ) {
			OpProject arg0 = (OpProject) op;
			//System.out.println("OpProject sub op:"+arg0.getSubOp());
			//System.out.println("Visited size (OpProject): "+visited.size());
			if(visited.get(arg0.getSubOp())==false) {
				walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);
			}
		}
		
		if(op instanceof OpOrder ) {
			OpOrder arg0 = (OpOrder) op;
			
			if(visited.get(arg0.getSubOp())==false) walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);
			
		}
		
		if(op instanceof OpList) {
			OpList arg0 = (OpList) op;
			
			if(visited.get(arg0.getSubOp())==false) walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);
		}
		
		if(op instanceof OpDisjunction) {
			OpDisjunction arg0 = (OpDisjunction) op;

			List<Op> ops = arg0.getElements();
			for(Op o:ops){
				if(visited.get(o)==false)
					walkAlgebraTreeRecursive(o, level + 1);
			}
			
		}
		
		if(op instanceof OpSequence) {
			OpSequence arg0 = (OpSequence) op;
			List<Op> ops = arg0.getElements();
			for(Op o:ops){
				if(visited.get(o)==false)
					walkAlgebraTreeRecursive(o, level + 1);
			}
			
		}
		
		if(op instanceof OpConditional) {
			OpConditional arg0 = (OpConditional) op;
			if(visited.get(arg0.getLeft())==false)	walkAlgebraTreeRecursive(arg0.getLeft(), level + 1);
			if(visited.get(arg0.getRight())==false)	walkAlgebraTreeRecursive(arg0.getRight(), level + 1);
			
			
		}
		
		if(op instanceof OpMinus) {
			OpMinus arg0 = (OpMinus) op;
			if(visited.get(arg0.getLeft())==false)	walkAlgebraTreeRecursive(arg0.getLeft(), level + 1);
			if(visited.get(arg0.getRight())==false)	walkAlgebraTreeRecursive(arg0.getRight(), level + 1);
			
		}
		
		if(op instanceof OpDiff) {
			OpDiff arg0 = (OpDiff) op;
			if(visited.get(arg0.getLeft())==false)	walkAlgebraTreeRecursive(arg0.getLeft(), level + 1);
			if(visited.get(arg0.getRight())==false)	walkAlgebraTreeRecursive(arg0.getRight(), level + 1);
			
		}
		
		if(op instanceof OpUnion) {
			OpUnion arg0 = (OpUnion) op;
			if(visited.get(arg0.getLeft())==false)	walkAlgebraTreeRecursive(arg0.getLeft(), level + 1);
			if(visited.get(arg0.getRight())==false)	walkAlgebraTreeRecursive(arg0.getRight(), level + 1);
			
		}
		
		if(op instanceof OpLeftJoin) {
			OpLeftJoin arg0 = (OpLeftJoin) op;
			if(visited.get(arg0.getLeft())==false)	walkAlgebraTreeRecursive(arg0.getLeft(), level + 1);
			if(visited.get(arg0.getRight())==false)	walkAlgebraTreeRecursive(arg0.getRight(), level + 1);
			
		}
		
		if(op instanceof OpJoin) {
			OpJoin arg0 = (OpJoin) op;
			if(visited.get(arg0.getLeft())==false)	walkAlgebraTreeRecursive(arg0.getLeft(), level + 1);
			if(visited.get(arg0.getRight())==false)	walkAlgebraTreeRecursive(arg0.getRight(), level + 1);
			
		}
		
		if(op instanceof OpExtend) {
			OpExtend arg0 = (OpExtend) op;
			if(visited.get(arg0.getSubOp())==false) walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);
			
		}
		
		if(op instanceof OpAssign) {
			OpAssign arg0 = (OpAssign) op;
			if(visited.get(arg0.getSubOp())==false) walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);
			
		}
		
		if(op instanceof OpLabel) {
			OpLabel arg0 = (OpLabel) op;
			if(visited.get(arg0.getSubOp())==false) walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);
			
		}
		
		if(op instanceof OpDatasetNames) {
			
			
			
		}
		
		if(op instanceof OpService) {
			OpService arg0 = (OpService) op;
			if(visited.get(arg0.getSubOp())==false) walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);
			
		}
		
		if(op instanceof OpGraph) {
			OpGraph arg0 = (OpGraph) op;
			if(visited.get(arg0.getSubOp())==false) walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);
			
		}
		
		if(op instanceof OpFilter) {
			OpFilter arg0 = (OpFilter) op;
			if(visited.get(arg0.getSubOp())==false) walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);
			
			
		}
		
		if(op instanceof OpPropFunc) {
			OpPropFunc arg0 = (OpPropFunc) op;
			if(visited.get(arg0.getSubOp())==false) walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);
			
		}
		
		if(op instanceof OpProcedure) {
			OpProcedure arg0 = (OpProcedure) op;
			if(visited.get(arg0.getSubOp())==false) walkAlgebraTreeRecursive(arg0.getSubOp(), level + 1);
		}
		
		if(op instanceof OpNull) {
			
			
		}
		
		if(op instanceof OpTable) {
			
			
			
		}
		
		if(op instanceof OpPath) {
			
			
			
		}
		
		if(op instanceof OpQuad ) {
			
			
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
	
	private void walkAlgebraTreeOld(Op op, final int level) {
		
		visited.put(op, true);
		System.out.println("Level:"+level);
		System.out.println(op);
		
		
		if(treeHeight<level) {
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
				System.out.println("OpProject sub op:"+arg0.getSubOp());
				System.out.println("Visited size (OpProject): "+visited.size());
				
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
				for(Op o:ops){
					
						walkAlgebraTreeOld(o, level + 1);
				}
				
			}
			
			public void visit(OpSequence arg0) {
				List<Op> ops = arg0.getElements();
				for(Op o:ops){
					
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
				
				
				
			}
		});
	}
	
	public static void test(String queryStr) {
		AlgebraFeatureExtractor fe = new AlgebraFeatureExtractor();
		
		//String queryStr = "PREFIX foaf:    <http://xmlns.com/foaf/0.1/> SELECT ?name ?email WHERE {  ?x foaf:knows ?y . ?y foaf:name ?name .  OPTIONAL { ?y foaf:mbox ?email }  } offset 2000 limit 1000";
		
//		String queryStr = "PREFIX : <http://example/>"+
//							"	SELECT ?x"+
//							"	{"+
//							"	  ?x a :Toy ."+
//							"	  { SELECT ?x ( count(?order) as ?q ) { ?x :order ?order. ?x :no ?y } GROUP BY ?x  limit 20}"+
//							"	  FILTER ( ?q > 5 )"+
//							"	} offset 100 limit 50";
							
							
		//# Find the types of :x, following subClassOf
		//String queryStr = "PREFIX :     <http://example/> PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>  SELECT * { ?s :p1 ?v1 OPTIONAL {?s :p2 ?v2 FILTER(?v1<3) } }";
		//String q = "/sparql/?query=SELECT+%3Fabstract+WHERE+{+%3Chttp%3A%2F%2Fdbpedia.org%2Fresource%2FHertfordshire%3E+%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2Fabstract%3E+%3Fabstract.+FILTER+langMatches(lang(%3Fabstract)%2C+%27en%27)+}&format=json";
		//String q = "/sparql?format=xml&default-graph-uri=http://dbpedia.org&query=%0APREFIX%20rdf%3A%20%3Chttp%3A%2F%2Fwww.w3.org%2F1999%2F02%2F22-rdf-syntax-ns%23%3E%0APREFIX%20rdfs%3A%20%3Chttp%3A%2F%2Fwww.w3.org%2F2000%2F01%2Frdf-schema%23%3E%0APREFIX%20yago%3A%20%20%20%20%3Chttp%3A%2F%2Fdbpedia.org%2Fclass%2Fyago%2F%3E%0APREFIX%20foaf%3A%20%3Chttp%3A%2F%2Fxmlns.com%2Ffoaf%2F0.1%2F%3E%0APREFIX%20p%3A%20%20%3Chttp%3A%2F%2Fdbpedia.org%2Fproperty%2F%3E%0APREFIX%20geo%3A%20%3Chttp%3A%2F%2Fwww.w3.org%2F2003%2F01%2Fgeo%2Fwgs84_pos%23%3E%0APREFIX%20skos%3A%20%3Chttp%3A%2F%2Fwww.w3.org%2F2004%2F02%2Fskos%2Fcore%23%3E%0A%0ASELECT%20%2A%20WHERE%20%7B%20%0A%20%20%20%20%20%20%3Chttp%3A%2F%2Fdbpedia.org%2Fresource%2FCategory%3AHand_to_waist_lifts%3E%20skos%3Abroader%20%3Fcategory%20.%0A%20%20%20%7D%0A";
		//String q = "/sparql?query=SELECT+DISTINCT+%3Fs+%0A%3Fontology_affiliation%0A%3Fontology_abstract%0A%3Fontology_campus%0A%3Fontology_chairman%0A%3Fontology_chancellor%0A%3Fontology_city%0A%3Fontology_country%0A%3Fontology_dean%0A%3Fontology_director%0A%3Fontology_endowment%0A%3Fontology_facultySize%0A%3Fontology_formerName%0A%3Fontology_foundingDate%0A%3Fontology_mascot%0A%3Fontology_motto%0A%3Fontology_numberOfDoctoralStudents%0A%3Fontology_numberOfPostgraduateStudents%0A%3Fontology_numberOfStudents%0A%3Fontology_numberOfUndergraduateStudents%0A%3Fontology_officialSchoolColour%0A%3Fontology_other%0A%3Fontology_president%0A%3Fontology_principal%0A%3Fontology_province%0A%3Fontology_rector%0A%3Fontology_sport%0A%3Fontology_staff%0A%3Fontology_state%0A%3Fontology_thumbnail%0A%3Fontology_logo%0A%3Fontology_type%0A%3Fontology_viceChancellor%0A%3Fproperty_academicDean%0A%3Fproperty_academicStaff%0A%3Fproperty_acronym%0A%3Fproperty_address%0A%3Fproperty_administrativeStaff%0A%3Fproperty_advert%0A%3Fproperty_affiliation%0A%3Fproperty_affiliations%0A%3Fproperty_afiliation%0A%3Fproperty_lat+%0A%3Fproperty_long%0A%3Fproperty_established%0AWHERE+%7B%0A%3Fs+a+%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2FEducationalInstitution%3E%2C+%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2FUniversity%3E+.%0A%3Fs+%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2Fcountry%3E+%3Chttp%3A%2F%2Fdbpedia.org%2Fresource%2FBrazil%3E+.%0A%3Fcountry+%3Chttp%3A%2F%2Fpurl.org%2Fdc%2Fterms%2Fsubject%3E+%3Chttp%3A%2F%2Fdbpedia.org%2Fresource%2FCategory%3ASouth_American_countries%3E+.%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2Faffiliation%3E+%3Fontology_affiliation+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2Fabstract%3E+%3Fontology_abstract+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2Fcampus%3E+%3Fontology_campus+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2Fchairman%3E+%3Fontology_chairman+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2Fchancellor%3E+%3Fontology_chancellor+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2Fcity%3E+%3Fontology_city+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2Fcountry%3E+%3Fontology_country+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2Fdean%3E+%3Fontology_dean+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2Fdirector%3E+%3Fontology_director+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2Fendowment%3E+%3Fontology_endowment+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2FfacultySize%3E+%3Fontology_facultySize+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2FformerName%3E+%3Fontology_formerName+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2FfoundingDate%3E+%3Fontology_foundingDate+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2Fmascot%3E+%3Fontology_mascot+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2Fmotto%3E+%3Fontology_motto+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2FnumberOfDoctoralStudents%3E+%3Fontology_numberOfDoctoralStudents+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2FnumberOfPostgraduateStudents%3E+%3Fontology_numberOfPostgraduateStudents+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2FnumberOfStudents%3E+%3Fontology_numberOfStudents+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2FnumberOfUndergraduateStudents%3E+%3Fontology_numberOfUndergraduateStudents+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2FofficialSchoolColour%3E+%3Fontology_officialSchoolColour+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2Fother%3E+%3Fontology_other+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2Fpresident%3E+%3Fontology_president+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2Fprincipal%3E+%3Fontology_principal+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2Fprovince%3E+%3Fontology_province+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2Frector%3E+%3Fontology_rector+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2Fsport%3E+%3Fontology_sport+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2Fstaff%3E+%3Fontology_staff+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2Fstate%3E+%3Fontology_state+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2Fthumbnail%3E+%3Fontology_thumbnail+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2Ftype%3E+%3Fontology_type+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2FviceChancellor%3E+%3Fontology_viceChancellor+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2FwikiPageExternalLink%3E+%3Fproperty_wikiPageExternalLink+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fproperty%2FacademicDean%3E+%3Fproperty_academicDean+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fproperty%2FacademicStaff%3E+%3Fproperty_academicStaff+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fproperty%2Facronym%3E+%3Fproperty_acronym+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fproperty%2Faddress%3E+%3Fproperty_address+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fproperty%2FadministrativeStaff%3E+%3Fproperty_administrativeStaff+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fproperty%2Fadvert%3E+%3Fproperty_advert+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fproperty%2Faffiliation%3E+%3Fproperty_affiliation+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fproperty%2Faffiliations%3E+%3Fproperty_affiliations+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fproperty%2Fafiliation%3E+%3Fproperty_afiliation+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fwww.w3.org%2F2003%2F01%2Fgeo%2Fwgs84_pos%23lat%3E+%3Fproperty_lat+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fwww.w3.org%2F2003%2F01%2Fgeo%2Fwgs84_pos%23long%3E+%3Fproperty_long+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fproperty%2Festablished%3E+%3Fproperty_established+.%7D%0AOPTIONAL+%7B%3Fs++%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2Flogo%3E+%3Fontology_logo+.%7D%0AFILTER+%28+langMatches%28lang%28%3Fontology_abstract%29%2C+%22es%22%29+%7C%7C+langMatches%28lang%28%3Fontology_abstract%29%2C+%22en%22%29+%29%0A%7D%0ALIMIT+4000%0A&results=json&output=json";
		//String queryStr = DBPediaUtils.getQueryForDBpedia(q);
		
		
		fe.setDebug(true);
		//String queryStr = "PREFIX foaf:       <http://xmlns.com/foaf/0.1/> SELECT DISTINCT ?name ?nick WHERE { ?x foaf:mbox <mailto:person@server.com> . ?x foaf:name ?name  OPTIONAL { ?x foaf:nick ?nick }}";
		if(queryStr==null) {
			queryStr = "PREFIX foaf:       <http://xmlns.com/foaf/0.1/> SELECT DISTINCT ?name ?nick WHERE { ?x foaf:mbox <mailto:person@server.com> . ?x foaf:name ?name  OPTIONAL { ?x foaf:nick ?nick }}";
			queryStr = "PREFIX umbel-sc: <http://umbel.org/umbel/sc/> PREFIX linkedgeodata: <http://linkedgeodata.org/triplify/> PREFIX dbpedia: <http://dbpedia.org/ontology/> PREFIX geonames: <http://www.geonames.org/ontology#> PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#> PREFIX owl: <http://www.w3.org/2002/07/owl#> PREFIX wgs84: <http://www.w3.org/2003/01/geo/wgs84_pos#> PREFIX foaf: <http://xmlns.com/foaf/0.1/> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> PREFIX lgdo: <http://linkedgeodata.org/ontology/> PREFIX dbpedia-owl: <http://dbpedia.org/ontology/> PREFIX lgdp: <http://linkedgeodata.org/property/>  PREFIX umbel-sc: <http://umbel.org/umbel/sc/> PREFIX linkedgeodata: <http://linkedgeodata.org/triplify/> PREFIX dbpedia: <http://dbpedia.org/ontology/> PREFIX geonames: <http://www.geonames.org/ontology#> PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#> PREFIX owl: <http://www.w3.org/2002/07/owl#> PREFIX wgs84: <http://www.w3.org/2003/01/geo/wgs84_pos#> PREFIX foaf: <http://xmlns.com/foaf/0.1/> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> PREFIX lgdo: <http://linkedgeodata.org/ontology/> PREFIX dbpedia-owl: <http://dbpedia.org/ontology/> PREFIX lgdp: <http://linkedgeodata.org/property/> SELECT DISTINCT ?b ?v0 FROM <http://linkedgeodata.org> WHERE { {?b a lgdo:City} UNION {?b a lgdo:Village} UNION {?b a lgdo:Suburb} UNION {?b a lgdo:Town}.  ?b wgs84:lat ?lat. ?b wgs84:long ?long. OPTIONAL { ?b rdfs:label ?v0 . } } OFFSET 708000 LIMIT 1000";
		}
		System.out.println(queryStr);
		double[] features = fe.extractFeatures(queryStr);
		
		for(double f:features) {
			System.out.print(" "+f);
		}
		
		int count = 0;
		System.out.println();
		List<Entry<String, Integer>> ll= new ArrayList<Entry<String, Integer>>(fe.featureIndex.entrySet());
		//fe.featureIndex.entrySet();
		
		Collections.sort(ll, new Comparator<Entry<String, Integer>>() {

			@Override
			public int compare(Entry<String, Integer> o1,
					Entry<String, Integer> o2) {
				// TODO Auto-generated method stub
				return o1.getValue()-o2.getValue();
			}
		});
		
		for(Entry<String, Integer> e:ll) {
			String header = e.getKey();
			if(count!=0) {
				System.out.print(",");
			}
			System.out.print(header);
			count++;
		}
	}
	public static void main(String[] args) {
		String queryStr=null;
		test(queryStr);
	}
}
