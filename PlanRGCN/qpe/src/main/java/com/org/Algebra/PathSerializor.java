package com.org.Algebra;

import java.io.PrintStream;

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

public class PathSerializor implements PathVisitor {
    PrintStream stream;
    public int indent = 0;
    private boolean notFirst = false; //whether predicate is the first or not

    private void print(String text){
        String indentStr = "";
        int t = indent;
        while (t > 0){
            indentStr +="  ";
            t--;
        }
        this.stream.println(indentStr+text);
    }
    public PathSerializor(PrintStream stream, int indent){
        this.stream = stream;
        this.indent = indent;

    }
    @Override
    public void visit(P_ZeroOrOne path) {
        path.getSubPath().visit(this);
    }

    @Override
    public void visit(P_ZeroOrMore1 path) {
        path.getSubPath().visit(this);
    }

    @Override
    public void visit(P_ZeroOrMoreN path) {
        path.getSubPath().visit(this);
    }
    
    @Override
    public void visit(P_OneOrMore1 path) {
        path.getSubPath().visit(this);
    }
    
    @Override
    public void visit(P_OneOrMoreN path) {
        path.getSubPath().visit(this);
    }
    
    @Override
    public void visit(P_NegPropSet pathNotOneOf) {
        for (P_Path0 i:pathNotOneOf.getNodes()){
            i.visit(this);
        }
    }
    @Override
    public void visit(P_Link pathNode) {
        // TODO Auto-generated method stub
        if(notFirst){
            print(", \""+pathNode.toString()+"\"");
        }else{
            print(" \""+pathNode.toString()+"\"");
            notFirst=true;
        }
        
    }
    @Override
    public void visit(P_Alt pathAlt) {
        pathAlt.getLeft().visit(this);
        pathAlt.getRight().visit(this);
    }
    
    @Override
    public void visit(P_Mod pathMod) {
        boolean notFirst2 = notFirst;
        notFirst = false;
        print("{\"max\":"+pathMod.getMax()+", \"min\": "+pathMod.getMin()+", \"Predicate\":");
        pathMod.getSubPath().visit(this);
        print("}");
        notFirst = notFirst2;
    }
    @Override
    public void visit(P_FixedLength pFixedLength) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'visit'");
    }
    @Override
    public void visit(P_Multi pathMulti) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'visit'");
    }
    @Override
    public void visit(P_ReverseLink pathNode) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'visit'");
    }


    @Override
    public void visit(P_Inverse inversePath) {
        inversePath.getSubPath().visit(this);
    }



    @Override
    public void visit(P_Distinct pathDistinct) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'visit'");
    }


    @Override
    public void visit(P_Shortest pathShortest) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'visit'");
    }


    @Override
    public void visit(P_Seq pathSeq) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'visit'");
    }
    
}
