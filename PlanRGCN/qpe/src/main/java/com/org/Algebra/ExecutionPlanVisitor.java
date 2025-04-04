package com.org.Algebra;


import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Iterator;

import org.apache.jena.graph.Triple;
import org.apache.jena.sparql.algebra.Op;
import org.apache.jena.sparql.algebra.OpVisitorByType;
import org.apache.jena.sparql.algebra.op.Op0;
import org.apache.jena.sparql.algebra.op.Op1;
import org.apache.jena.sparql.algebra.op.Op2;
import org.apache.jena.sparql.algebra.op.OpBGP;
import org.apache.jena.sparql.algebra.op.OpFilter;
import org.apache.jena.sparql.algebra.op.OpLeftJoin;
import org.apache.jena.sparql.algebra.op.OpN;
import org.apache.jena.sparql.algebra.op.OpPath;
import org.apache.jena.sparql.algebra.op.OpProcedure;
import org.apache.jena.sparql.algebra.op.OpPropFunc;
import org.apache.jena.sparql.algebra.op.OpSequence;
import org.apache.jena.sparql.function.library.leviathan.degreesToRadians;
import org.apache.jena.sparql.path.P_Alt;
import org.apache.jena.sparql.path.P_Mod;
import org.apache.jena.sparql.path.P_NegPropSet;
import org.apache.jena.sparql.path.P_OneOrMore1;
import org.apache.jena.sparql.path.P_ZeroOrMore1;
import org.apache.jena.sparql.path.P_ZeroOrOne;
import org.apache.jena.sparql.path.Path;
import org.apache.jena.graph.Node;


/*
 * Output format
 * {
 *  operator name : "name"
 * child : [ {}]
 * }
 * 
 * 
 */

public class ExecutionPlanVisitor extends OpVisitorByType {
    PrintStream stream;
    public int indent = 0;

    private void print(String text){
        String indentStr = "";
        int t = indent;
        while (t > 0){
            indentStr +="  ";
            t--;
        }
        this.stream.println(indentStr+text);
    }
    private void openScope(){
        indent++;
    }
    private void closeScope(){
        indent--;
    }

    public ExecutionPlanVisitor(){
        this.stream = System.out;

    }

    public ExecutionPlanVisitor(boolean diable) {
        if (!diable) {
            this.stream = new PrintStream(new OutputStream() {
                public void write(int b) {
                    // DO NOTHING
                }
            });
        } else {
            this.stream = System.out;
        }

    }

    public ExecutionPlanVisitor(String path) throws FileNotFoundException{
        
        this.stream = new PrintStream(new FileOutputStream(path));
        
    }

    public void visit(Triple triple){
        String t = "{\"Subject\": \""+triple.getSubject().toString(true);
        t += "\", \"Predicate\": \""+triple.getPredicate().toString(true);
        
        String objectString;
        if (triple.getObject().isLiteral()){
            StringBuilder objBuilder = new StringBuilder();
            objBuilder.append('{');
            objBuilder.append('"');
            objBuilder.append("value")
            .append('"')
            .append(":")
            .append('"')
            .append(triple.getObject().getLiteralValue().toString())
            .append('"')
            .append(',')
            .append('"')
            .append("datatype")
            .append('"')
            .append(":")
            .append('"')
            .append(triple.getObject().getLiteralDatatypeURI().toString())
            .append('"')
            .append(',')

            .append('"')
            .append("langTag")
            .append('"')
            .append(":")
            .append('"')
            .append(triple.getObject().getLiteralLanguage())
            .append('"');
            
            objBuilder.append('}');
            objectString = objBuilder.toString();
            //t+= "\", \"Object\": \""+temp;
            //triple.getObject().getLiteralDatatype().
        }else{
            StringBuilder objBuilder = new StringBuilder();
            objBuilder.append('{');
            objBuilder.append('"');
            objBuilder.append("value")
            .append('"')
            .append(":")
            .append('"')
            .append(triple.getObject())
            .append('"');
            
            objBuilder.append('}');
            objectString = objBuilder.toString();
        }
        t += "\", \"Object\": "+objectString;
        t+= ", \"opName\": \"Triple\"}";
        print(t);
    }

    @Override 
    public void visit(OpBGP opBGP)                   {
        print("{\"opName\":\"BGP\", \"subOp\": [");
        openScope();
        Iterator<Triple> iter= opBGP.getPattern().iterator();
        while(iter.hasNext()){
            visit(iter.next());
            if (iter.hasNext())
                print(",");
        }
        closeScope();
        print("]}");
    }
    
    public void visit(Node n){
        print("Beginning node");
        //print(n.getName());
        print("Ending Node");

    }

    @Override 
    public void visit(OpPath opPath)                  {
        
        StringBuilder t = new StringBuilder();
        t.append("{\"opName\": \"")
        .append(opPath.getName())
        .append('"')
        .append(", \"Subject\": \"")
        .append(opPath.getTriplePath().getSubject().toString(true))
        .append('"')
        //.append(", \"Object\": \"")
        ;
        String objectString;
        if (opPath.getTriplePath().getObject().isLiteral()){
            StringBuilder objBuilder = new StringBuilder();
            objBuilder.append('{');
            objBuilder.append('"');
            objBuilder.append("value")
            .append('"')
            .append(":")
            .append('"')
            .append(opPath.getTriplePath().getObject().getLiteralValue().toString())
            .append('"')
            .append(',')
            .append('"')
            .append("datatype")
            .append('"')
            .append(":")
            .append('"')
            .append(opPath.getTriplePath().getObject().getLiteralDatatypeURI().toString())
            .append('"')
            .append(',')

            .append('"')
            .append("langTag")
            .append('"')
            .append(":")
            .append('"')
            .append(opPath.getTriplePath().getObject().getLiteralLanguage())
            .append('"');
            
            objBuilder.append('}');
            objectString = objBuilder.toString();
            //t+= "\", \"Object\": \""+temp;
            //triple.getObject().getLiteralDatatype().
        }else{
            StringBuilder objBuilder = new StringBuilder();
            objBuilder.append('{');
            objBuilder.append('"');
            objBuilder.append("value")
            .append('"')
            .append(":")
            .append('"')
            .append(opPath.getTriplePath().getObject())
            .append('"');
            
            objBuilder.append('}');
            objectString = objBuilder.toString();
        }
        t.append(", \"Object\": "+objectString);
        
        print(t.toString());
        Path path = opPath.getTriplePath().getPath();
        PathComplexityChecker compVisitor = new PathComplexityChecker();
        path.visit(compVisitor);
        boolean complex = false;
        if (compVisitor.getPathOperations().size() > 0){
            ArrayList<String> pathOps = compVisitor.getPathOperations();
            pathOps.remove("P_Inverse");
            if (pathOps.size() > 1)
                complex = true;
            print(", \"pathComplexity\": [");
            Iterator<String> iter = compVisitor.getPathOperations().iterator();
            while(iter.hasNext()){
                String pathtype= iter.next();
                print("\""+pathtype+"\"");
                if (iter.hasNext())
                print(", ");
            }
            print("]\n");
        }
        if (!complex){
            PathSerializor p = new PathSerializor(stream,indent);
            if (path instanceof P_Alt) {
                /*print(", \"pathComplexity\": [\"Addede to be filtered aways\"");
                print("],");*/
                print(", \"pathType\": \"alternative\"");
                print(",\"Predicates\":[");
                print("\"Not Implemented for now\"]");
                /*path.visit(p);
                print("]");*/
                print("}");
            } /*else if (path instanceof P_Multi) {
                print(", \"pathType\": \"multi\"");
                print(",\"Predicates\":[");
                path.visit(p);
                print(",\"Predicates\":[");
                path.visit(p);
                print("]");
                print("}");
            }*/else if (path instanceof P_ZeroOrOne) {
                print(", \"pathType\": \"zeroOrOne\"");
                print(",\"Predicates\":[");
                path.visit(p);
                print("]");
                print("}");
            }else if (path instanceof P_ZeroOrMore1) {
                print(", \"pathType\": \"ZeroOrMore\"");
                print(",\"Predicates\":[");
                path.visit(p);
                print("]");
                print("}");
            }else if (path instanceof P_OneOrMore1) {
                print(", \"pathType\": \"OneOrMore\"");
                print(",\"Predicates\":[");
                path.visit(p);
                print("]");
                print("}");
            }else if (path instanceof P_NegPropSet) {
                print(", \"pathType\": \"NotOneOf\"");
                print(",\"Predicates\":[");
                path.visit(p);
                print("]");
                print("}");
            }else if (path instanceof P_Mod) {
                print(", \"pathType\": \"Mod\"");
                print(",\"Predicates\":[");
                path.visit(p);
                print("]");
                print("}");
            }else{
                print(", \"pathType\": \""+path.getClass().toString()+"\"");
                //print(", \"pathComplexity\": [\"Addede to be filtered aways (unsupperted operator\"");
                //print("],");
                print("}");
                
            }
        }else{
            print("}");
        }
    }
    
    @Override 
    public void visit(OpSequence opSequence)          {
        print("{\"opName\": \""+opSequence.getName()+ "\", \"subOp\": [");
        openScope();
        //List<Op> lst = opSequence.getElements();
        Iterator<Op> iter = opSequence.iterator();
        while(iter.hasNext()){
            Op t = iter.next();
            openScope();
            t.visit(this);
            if (iter.hasNext()){
                print(",");
            }
            closeScope();
            
        }
        /* 
        for (Op t : lst){
            
            openScope();
            t.visit(this);
            print(",");
            closeScope();
        }*/
        closeScope();
        print("]}");
    }
    
    
    @Override 
    public void visit(OpFilter opFilter)              {
        //opFilter.getExprs().iterator();
        print("{\"opName\": \""+opFilter.getName()+"\" , "+ "\"expr\": \" "+ opFilter.getExprs().toString().replace("\"", "\\\"").replace("\n"," ")+"\", "+"\"subOp\": [");
        openScope();
        //TODO: add expressions to queryplan
        opFilter.getSubOp().visit(this);
        closeScope();
        print("]}");
    }
    
    @Override
    protected void visitN(OpN op) {
        print("{\"opName\": \""+op.getName()+ "\", \"subOp\": [");
        openScope();
        Iterator<Op> iter =  op.getElements().iterator();
        while(iter.hasNext()){
            Op t = iter.next();
            t.visit(this);
            if (iter.hasNext())
                print(",");
        }
        closeScope();
        print("]}");
        // TODO Auto-generated method stub
        //throw new UnsupportedOperationException("Unimplemented method 'visitN'");
    }
    @Override
    protected void visit2(Op2 op) {
        print("{\"opName\": \""+op.getName()+ "\", \"subOp\": [");
        openScope();
        //List<Op> lst = opSequence.getElements();
        op.getLeft().visit(this);
        print(",");
        op.getRight().visit(this);
        closeScope();
        print("]}");
        
        //op.getLeft().visit(this);
        //op.getRight().visit(this);
        // TODO Auto-generated method stub
        //throw new UnsupportedOperationException("Unimplemented method 'visit2'");
    }
    @Override 
    public void visit(OpProcedure opProc)             {
        print("{\"opName\": \""+opProc.getName()+ "\", \"subOp\": []}");
        //print("opProc");
        
    }
    
    @Override 
    public void visit(OpPropFunc opPropFunc)          {
        print("{\"opName\": \""+opPropFunc.getName()+ "\", \"subOp\": []}");
        //print("OpPropFunc");

    }

    /*@Override 
    public void visit(OpJoin opJoin)                  {
        opJoin.getRight()

    }*/



    
    @Override
    protected void visit1(Op1 op) {
        print("{\"opName\": \""+op.getName()+ "\", \"subOp\": [");
        openScope();
        //List<Op> lst = opSequence.getElements();
        op.getSubOp().visit(this);
        closeScope();
        print("]}");
        
        // TODO Auto-generated method stub
        //throw new UnsupportedOperationException("Unimplemented method 'visit1'");
    }
    @Override
    protected void visit0(Op0 op) {
        print("{\"opName\": \""+op.getName()+ "\", \"subOp\": []}");
        // TODO Auto-generated method stub
        //throw new UnsupportedOperationException("Unimplemented method 'visit0'");
    }
    @Override
    protected void visitLeftJoin(OpLeftJoin op) {
        print("{\"opName\": \""+op.getName()+ "\", \"subOp\": [");
        openScope();
        //List<Op> lst = opSequence.getElements();
        op.getLeft().visit(this);
        print(",");
        op.getRight().visit(this);
        closeScope();
        print("]}");

        //op.getLeft().visit(this);
        //op.getRight().visit(this);
        // TODO Auto-generated method stub
        //throw new UnsupportedOperationException("Unimplemented method 'visitLeftJoin'");
    }
    @Override
    protected void visitFilter(OpFilter op) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'visitFilter'");
    }
    
}
