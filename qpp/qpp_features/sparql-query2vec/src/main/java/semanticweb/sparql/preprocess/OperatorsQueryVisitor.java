package semanticweb.sparql.preprocess;

import org.apache.jena.query.Query;
import org.apache.jena.query.QueryFactory;
import org.apache.jena.sparql.algebra.*;
import org.apache.jena.sparql.algebra.op.*;
import org.apache.jena.sparql.path.*;

import java.util.HashMap;
import java.util.Map;
import java.util.Vector;

public class OperatorsQueryVisitor  extends OpVisitorBase {
    private Map<String, Integer> featureIndex;
    final double[] features;

    public OperatorsQueryVisitor() {
        super();
        this.featureIndex = new HashMap<>();
        this.features = new double[featureIndex.size()];

    }
    public OperatorsQueryVisitor(Map<String, Integer> featureIndex) {
        super();
        this.featureIndex = featureIndex;
        this.features = new double[featureIndex.size()];

    }

    public double[] getFeatures() {
        return features;
    }

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
//        System.out.println("join ");
        features[featureIndex.get("join")] += 1.0;
    }

    public void visit(OpLeftJoin opleftJoin) {
//        System.out.println("leftjoin ");
        features[featureIndex.get("leftjoin")] += 1.0;

    }

    public void visit(OpUnion opUnion) {
//        System.out.println("union ");
        features[featureIndex.get("union")] += 1.0;

    }

    public void visit(OpFilter opFilter) {
//        System.out.println("filter ");
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

//        System.out.println("group ");
        features[featureIndex.get("group")] += 1.0;

    }

    public void visit(OpAssign opAssign) {

//        System.out.println("assign ");
        features[featureIndex.get("assign")] += 1.0;

    }

    public void visit(OpSequence opSequence) {

//        System.out.println("sequence ");
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
    public static void main(String args[]) {
        String queryFile = "/home/daniel/Desktop/query";
        Query query = QueryFactory.read(queryFile);
        Op op = (new AlgebraGenerator()).compile(query);
        OperatorsQueryVisitor vcs = new OperatorsQueryVisitor();
        OpWalker ow = new OpWalker();
        ow.walk(op, vcs);
//        Vector<Op> ss = vcs.getServices();
//        System.out.println(ss);
    }
}
