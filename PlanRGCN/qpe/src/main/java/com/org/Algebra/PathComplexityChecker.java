package com.org.Algebra;
import java.io.PrintStream;
import java.util.ArrayList;

import org.apache.jena.sparql.path.P_Alt;
import org.apache.jena.sparql.path.P_Distinct;
import org.apache.jena.sparql.path.P_FixedLength;
import org.apache.jena.sparql.path.P_Inverse;
import org.apache.jena.sparql.path.P_Link;
import org.apache.jena.sparql.path.P_Mod;
import org.apache.jena.sparql.path.P_Multi;
import org.apache.jena.sparql.path.P_NegPropSet;
import org.apache.jena.sparql.path.P_OneOrMore1;
import org.apache.jena.sparql.path.P_OneOrMoreN;
import org.apache.jena.sparql.path.P_Path0;
import org.apache.jena.sparql.path.P_ReverseLink;
import org.apache.jena.sparql.path.P_Seq;
import org.apache.jena.sparql.path.P_Shortest;
import org.apache.jena.sparql.path.P_ZeroOrMore1;
import org.apache.jena.sparql.path.P_ZeroOrMoreN;
import org.apache.jena.sparql.path.P_ZeroOrOne;
import org.apache.jena.sparql.path.PathVisitor;

public class PathComplexityChecker implements PathVisitor {
    public ArrayList<String> pathOperations = new ArrayList<>();

    public ArrayList<String> getPathOperations() {
        return pathOperations;
    }

    @Override
    public void visit(P_ZeroOrOne path) {
        pathOperations.add(path.getClass().getSimpleName());
        path.getSubPath().visit(this);
    }
    
    @Override
    public void visit(P_ZeroOrMore1 path) {
        pathOperations.add(path.getClass().getSimpleName());
        path.getSubPath().visit(this);
    }

    @Override
    public void visit(P_ZeroOrMoreN path) {
        pathOperations.add(path.getClass().getSimpleName());
        path.getSubPath().visit(this);
    }
    
    @Override
    public void visit(P_OneOrMore1 path) {
        pathOperations.add(path.getClass().getSimpleName());
        path.getSubPath().visit(this);
    }
    
    @Override
    public void visit(P_OneOrMoreN path) {
        pathOperations.add(path.getClass().getSimpleName());
        path.getSubPath().visit(this);
    }
    
    @Override
    public void visit(P_NegPropSet pathNotOneOf) {
        pathOperations.add(pathNotOneOf.getClass().getSimpleName());
        for (P_Path0 i:pathNotOneOf.getNodes()){
            i.visit(this);
        }
    }
    @Override
    public void visit(P_Link pathNode) { 
    }
    @Override
    public void visit(P_Alt pathAlt) {
        pathOperations.add(pathAlt.getClass().getSimpleName());

        pathAlt.getLeft().visit(this);
        pathAlt.getRight().visit(this);
    }
    
    @Override
    public void visit(P_FixedLength pFixedLength) {
        pathOperations.add(pFixedLength.getClass().getSimpleName());
        pFixedLength.getSubPath().visit(this);
    }
    @Override
    public void visit(P_Multi pathMulti) {
        // TODO Auto-generated method stub
        pathOperations.add(pathMulti.getClass().getSimpleName());
        pathMulti.getSubPath().visit(this);
    }
    @Override
    public void visit(P_ReverseLink pathNode) {
        pathOperations.add(pathNode.getClass().getSimpleName());
        
    }
    
    
    @Override
    public void visit(P_Inverse inversePath) {
        // TODO Auto-generated method stub
        pathOperations.add(inversePath.getClass().getSimpleName());
        inversePath.getSubPath().visit(this);
    }

    @Override
    public void visit(P_Mod pathMod) {
        pathOperations.add(pathMod.getClass().getSimpleName());
        pathMod.getSubPath().visit(this);
    }


    @Override
    public void visit(P_Distinct pathDistinct) {
        pathOperations.add(pathDistinct.getClass().getSimpleName());
        pathDistinct.getSubPath().visit(this);
    }


    @Override
    public void visit(P_Shortest pathShortest) {
        pathOperations.add(pathShortest.getClass().getSimpleName());
        pathShortest.getSubPath().visit(this);
    }
    
    
    @Override
    public void visit(P_Seq pathSeq) {
        pathOperations.add(pathSeq.getClass().getSimpleName());
        pathSeq.getLeft().visit(this);
        pathSeq.getRight().visit(this);
    }
    
}
