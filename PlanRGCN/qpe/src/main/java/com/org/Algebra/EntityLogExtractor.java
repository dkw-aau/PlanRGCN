package com.org.Algebra;

import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Set;

import org.apache.jena.query.Query;
import org.apache.jena.query.QueryFactory;
import org.apache.jena.sparql.algebra.Algebra;
import org.apache.jena.sparql.algebra.Op;
import org.apache.jena.sparql.algebra.OpVisitorByTypeBase;
import org.apache.jena.sparql.algebra.op.Op0;
import org.apache.jena.sparql.algebra.op.Op1;
import org.apache.jena.sparql.algebra.op.Op2;
import org.apache.jena.sparql.algebra.op.OpBGP;
import org.apache.jena.sparql.algebra.op.OpExt;
import org.apache.jena.sparql.algebra.op.OpFilter;
import org.apache.jena.sparql.algebra.op.OpLeftJoin;
import org.apache.jena.sparql.algebra.op.OpN;

import com.google.gson.Gson;
import com.google.gson.JsonIOException;
import com.org.QueryReader.LSQreader;

import org.apache.jena.graph.Triple;

public class EntityLogExtractor {
    Set<String> entities = new HashSet<String>();
    ArrayList<String> badQueries = new ArrayList<String>();
    public ArrayList<String> getBadQueries() {
        return badQueries;
    }

    public Set<String> getEntities() {
        return entities;
    }

    public void extractEntities(String query){
        
        
            Query q = QueryFactory.create(query);
            Op o = Algebra.compile(q);
            o.visit(new OpVisitorByTypeBase(){
                public void visit(OpBGP bgp){
                    for (Triple t: bgp.getPattern().getList()){
                        if (t.getSubject().isURI())
                            entities.add(t.getSubject().toString());
                        if (t.getObject().isURI())
                            entities.add(t.getObject().toString());
                    }
                }
                @Override
                public void visitN(OpN op) {
                    for(Op i : op.getElements()){
                        i.visit(this);
                    }
                }
                
                @Override
                public void visit2(Op2 op) {
                    op.getLeft().visit(this);
                    op.getRight().visit(this);
                }
                
                @Override
                public void visit1(Op1 op) {
                    op.getSubOp().visit(this);
                }
                
                @Override
                public void visit0(Op0 op) {
                }   
                
                @Override
                public void visitExt(OpExt op) {
                }    
                
                @Override
                public void visitFilter(OpFilter op) {
                    op.getSubOp().visit(this);
                }    
                
                @Override
                public void visitLeftJoin(OpLeftJoin op) { 
                    op.getLeft().visit(this);
                    op.getRight().visit(this);
                }
            });
        
            
    }
    
    public void extractEntities(LinkedList<String> queries){
        Iterator<String> iter = queries.iterator();
        
        while(iter.hasNext()){
            String q = iter.next();
            try{
                this.extractEntities(q);
            }catch(org.apache.jena.query.QueryException e){
                System.out.println("query parsing error");
                e.printStackTrace();
                badQueries.add(q);
            }catch(org.apache.jena.riot.RiotException e){
                System.out.println("query parsing error");
                e.printStackTrace();
                badQueries.add(q);
            }
        }
    }

    public void save( String outputPath) throws JsonIOException, IOException{
        FileWriter writer = new FileWriter(outputPath);
        Gson gson = new Gson();
        gson.toJson(this.entities,writer);
        writer.flush();
        writer.close();

    }

    public static void run(String queryLogPath, String predicateSavePath){
        try {
                    LSQreader reader = new LSQreader(queryLogPath);
                    LinkedList<String> queries = reader.getQueries();
                    EntityLogExtractor extractor = new EntityLogExtractor();
                    
                    extractor.extractEntities(queries);
                    System.out.println("Amount of distinct predicates in dataset: "+extractor.getEntities().size());
                    System.out.println("Amount of illegal queries in dataset: "+extractor.badQueries.size());
                    extractor.save( predicateSavePath);
        } catch (FileNotFoundException e) {
            System.out.println("Could not read file " + queryLogPath);
            e.printStackTrace();
        }catch(JsonIOException|IOException e){
            System.out.println("Write destination is not available!! "+predicateSavePath);
            e.printStackTrace();
        }
    }
}
