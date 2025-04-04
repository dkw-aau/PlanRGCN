package com.org.ex_plan;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;

import org.apache.jena.graph.Triple;
import org.apache.jena.sparql.algebra.Op;
import org.apache.jena.sparql.algebra.OpVisitorByType;
import org.apache.jena.sparql.algebra.op.Op0;
import org.apache.jena.sparql.algebra.op.Op1;
import org.apache.jena.sparql.algebra.op.Op2;
import org.apache.jena.sparql.algebra.op.OpBGP;
import org.apache.jena.sparql.algebra.op.OpConditional;
import org.apache.jena.sparql.algebra.op.OpFilter;
import org.apache.jena.sparql.algebra.op.OpLeftJoin;
import org.apache.jena.sparql.algebra.op.OpN;
import org.apache.jena.sparql.algebra.op.OpPath;
import org.apache.jena.sparql.algebra.op.OpProcedure;
import org.apache.jena.sparql.algebra.op.OpPropFunc;
import org.apache.jena.sparql.algebra.op.OpSequence;
import org.apache.jena.sparql.core.Var;
import org.apache.jena.sparql.expr.Expr;
import org.apache.jena.sparql.algebra.op.OpProject;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

public class QueryGraphVisitor extends OpVisitorByType {
    ArrayList<com.org.ex_plan.Node> nodes = new ArrayList<>();
    public ArrayList<Edge> edges = new ArrayList<>();
    
    //Auxilary datastructure for processing different things
    public HashMap<String, LinkedList<com.org.ex_plan.Node>> joinableTermMap = new HashMap<>();
    public HashSet<BinaryOperator> binOps = new HashSet<>();
    public ArrayList<com.org.ex_plan.Node> filters = new ArrayList<>();


    public QueryGraphVisitor(){

    }
    public void add_tp_edges(String key, TP node){
        if(joinableTermMap.keySet().contains(key)){
            for(com.org.ex_plan.Node i: joinableTermMap.get(key)){
                String edgeType = "";
                edgeType += i.getJoinedVar(key);
                edgeType += ("_"+node.getJoinedVar(key));
                edges.add(new Edge(i, node, edgeType));
            }
        }else{
            joinableTermMap.put(key, new LinkedList<>());
        }
        joinableTermMap.get(key).add(node);
    }

    public void addBinaryOPForTP(com.org.ex_plan.Node node){
        Iterator<BinaryOperator> iter = binOps.iterator();
        while(iter.hasNext()){
            BinaryOperator binOp = iter.next();
            binOp.addNode(node);
        }
    }

    public void visit(Triple triple){
        TP tp = new TP(triple.getSubject().toString(true),triple.getPredicate().toString(true));
        tp.isLiteral = triple.getObject().isLiteral();
        if (tp.isLiteral){
            tp.object = triple.getObject().getLiteralValue().toString();
            tp.objectDatatype = triple.getObject().getLiteralDatatypeURI().toString();
            tp.objectLang =  triple.getObject().getLiteralLanguage();
        }else{
            tp.object = triple.getObject().toString();
        }
        addBinaryOPForTP(tp);
        add_tp_edges(tp.subject, tp);
        add_tp_edges(tp.predicate, tp);
        add_tp_edges(tp.object, tp);
        nodes.add(tp);
    }

    @Override 
    public void visit(OpBGP opBGP)                   {
        Iterator<Triple> iter= opBGP.getPattern().iterator();
        while(iter.hasNext()){
            visit(iter.next());
        }
    }

    @Override 
    public void visit(OpPath opPath)                  {
        PP pp = new PP(opPath.getTriplePath().getSubject().toString(true));
        pp.isLiteral = opPath.getTriplePath().getObject().isLiteral();
        if (pp.isLiteral){
            pp.object = opPath.getTriplePath().getObject().getLiteralValue().toString();
            pp.objectDatatype = opPath.getTriplePath().getObject().getLiteralDatatypeURI().toString();
            pp.objectLang =  opPath.getTriplePath().getObject().getLiteralLanguage();
        }else{
            pp.object = opPath.getTriplePath().getObject().toString();
        }
        PathComplexityChecker ppVisitor = new PathComplexityChecker();
        opPath.getTriplePath().getPath().visit(ppVisitor);
        
        pp.pred_max = ppVisitor.mod_max;
        pp.pred_min = ppVisitor.mod_min;
        pp.pathComplexity = ppVisitor.getPathOperations();
        pp.predicateList = ppVisitor.predicates;

        addBinaryOPForTP(pp);
        add_tp_edges(pp.subject, pp);
        for (String pred : pp.predicateList){
            add_tp_edges(pred, pp);
        }
        add_tp_edges(pp.object, pp);
        nodes.add(pp);
        //Gson gson = new GsonBuilder().disableHtmlEscaping().create();
    }
    
    @Override 
    public void visit(OpSequence opSequence)          {
        Iterator<Op> iter = opSequence.iterator();
        while(iter.hasNext()){
            Op t = iter.next();
            t.visit(this);
            
        }
    }
    
    
    @Override 
    public void visit(OpFilter opFilter)              {
        //print("{\"opName\": \""+opFilter.getName()+"\" , "+ "\"expr\": \" "+ opFilter.getExprs().toString().replace("\"", "\\\"").replace("\n"," ")+"\", "+"\"subOp\": [");
        FilterNode filt = new FilterNode(opFilter.getExprs().toString());
        HashSet<String> varNames = new HashSet<>();
        for( Expr i : opFilter.getExprs()){
                for (Var varName : i.getVarsMentioned()){
                    varNames.add(varName.toString());
                }
        }
        filt.addVars(varNames);
        filters.add(filt);

        opFilter.getSubOp().visit(this);
    }
    
    @Override
    protected void visitN(OpN op) {
        Iterator<Op> iter =  op.getElements().iterator();
        while(iter.hasNext()){
            Op t = iter.next();
            t.visit(this);
        }
    }
    @Override
    protected void visit2(Op2 op) {
        //print("{\"opName\": \""+op.getName()+ "\", \"subOp\": [");
        //System.out.println(op.getName());
        op.getLeft().visit(this);
        op.getRight().visit(this);
    }
    @Override 
    public void visit(OpProcedure opProc)             {
        //print("{\"opName\": \""+opProc.getName()+ "\", \"subOp\": []}");
        //print("opProc");
        
    }
    
    @Override 
    public void visit(OpPropFunc opPropFunc)          {
        //System.out.println("Func  "+opPropFunc.getName());
        //print("{\"opName\": \""+opPropFunc.getName()+ "\", \"subOp\": []}");
        //print("OpPropFunc");

    }

    @Override
    protected void visit1(Op1 op) {
        op.getSubOp().visit(this);
    }
    
    @Override 
    public void visit(OpProject op)                   {
        
        /*
         * If we want variable in projections
        HashSet<String> varNames = new HashSet<>();
        for( Expr i : opFilter.getExprs()){
                for( Var varname : op.getVars()){}
                    varNames.add(varName.toString());
                }
        }
         */
        op.getSubOp().visit(this);
    }

    @Override
    protected void visit0(Op0 op) {
    }

    @Override
    protected void visitLeftJoin(OpLeftJoin op) {
        BinaryOperator leftjoin = new BinaryOperator("Optional");
        binOps.add(leftjoin);
        op.getLeft().visit(this);
        leftjoin.startRight();
        op.getRight().visit(this);
        leftjoin.done();
    }

    // will not be visited even with Filter 
    @Override
    protected void visitFilter(OpFilter op) {
        //System.out.println("Filter here");
    }

    public void assignNodeIds(){
        int id = 0;
        for(com.org.ex_plan.Node i: nodes){
            i.nodeId = id;
            id++;
        }
    }
    public void add_bin_edges() {
        for(BinaryOperator i : binOps){
            edges.addAll(i.edges);
        }
    }

    public void add_filters(){
        for(com.org.ex_plan.Node filt : filters){
            for(String key : filt.getTerms()){
                if(joinableTermMap.keySet().contains(key)){
                    for(com.org.ex_plan.Node i: joinableTermMap.get(key)){
                        edges.add(new Edge(filt, i, "filter"));
                    }
                }
            }
            nodes.add(filt);
        }
    }

    public void add_single_tp_edges(){
        HashSet<Node> edge_nodes = new HashSet<>();
        for (Edge e: edges){
            edge_nodes.add(e.node1);
            edge_nodes.add(e.node2);
        }
        for (Node n: nodes){
            if(!edge_nodes.contains(n)){
                edges.add(new Edge(n, n, "SingleTripleOrCatesian"));
            }
        }
        
    }
    
    public void finalize_edges(){
        add_filters();
        assignNodeIds();
        add_bin_edges();
        add_single_tp_edges();
    }

    public String edgeList(){
        Gson gson = new GsonBuilder().enableComplexMapKeySerialization().disableHtmlEscaping().create();
        ArrayList<ArrayList<Object>> edgeList = new ArrayList<>();
        for (Edge e: edges){
            ArrayList<Object> edge = new ArrayList<>();
            edge.add(e.node1.nodeId);
            edge.add(e.node2.nodeId);
            edge.add(e.edge_type);
            edgeList.add(edge);
        }
        return gson.toJson(edgeList);
    }

    public String nodeList(){
        Gson gson = new GsonBuilder().enableComplexMapKeySerialization().disableHtmlEscaping().create();
        return gson.toJson(nodes);
    }

    //Graph in terms of an relational edge list and a node set.
    public String graph(){
        Gson gson = new GsonBuilder().enableComplexMapKeySerialization().disableHtmlEscaping().create();
        ArrayList<ArrayList<Object>> edgeList = new ArrayList<>();
        for (Edge e: edges){
            ArrayList<Object> edge = new ArrayList<>();
            edge.add(e.node1.nodeId);
            edge.add(e.node2.nodeId);
            edge.add(e.edge_type);
            edgeList.add(edge);
        }

        HashMap<String, ArrayList> mapper = new HashMap<>();
        mapper.put("edges", edgeList);
        mapper.put("nodes", nodes);

        return gson.toJson(mapper);
    }
    @Override
    public void visit(OpConditional op) {
        BinaryOperator leftjoin = new BinaryOperator("Optional");
        binOps.add(leftjoin);
        op.getLeft().visit(this);
        leftjoin.startRight();
        op.getRight().visit(this);
        leftjoin.done();
    }
    
}
