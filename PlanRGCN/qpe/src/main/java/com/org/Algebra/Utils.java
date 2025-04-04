package com.org.Algebra;

import org.apache.jena.sparql.algebra.*;
import org.apache.jena.sparql.sse.SSE;

import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;
import com.opencsv.CSVWriter;
import com.org.QueryReader.LSQreader;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.concurrent.TimeUnit;

import org.apache.commons.lang3.time.StopWatch;
import org.apache.jena.query.Query;
import org.apache.jena.query.QueryFactory;

public class Utils {
    public static boolean sub_id = false; 

    public void create_algebra_test(String query) {
        Query q = QueryFactory.create(query);
        Op o = Algebra.compile(q);
        o = Algebra.optimize(o);
        SSE.write(o);
        try {
            ExecutionPlanVisitor visitor = new ExecutionPlanVisitor("/PlanRGCN/test.json");
            // o.visit(visitor);
            // o.visit(new CustomWalker(visitor));
            o.visit(visitor);
            // OpWalker.walk(o, visitor);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    public void create_algebra_singleQuery(String query, String output) {
        Query q = QueryFactory.create(query);
        Op o = Algebra.compile(q);
        o = Algebra.optimize(o);
        SSE.write(o);
        try {
            ExecutionPlanVisitor visitor = new ExecutionPlanVisitor(output);
            // o.visit(visitor);
            // o.visit(new CustomWalker(visitor));
            o.visit(visitor);
            // OpWalker.walk(o, visitor);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    public void extract_query_plan(String query, String filepath) {
        Query q = QueryFactory.create(query);
        Op o = Algebra.compile(q);
        o = Algebra.optimize(o);
        //SSE.write(o);
        try {
            ExecutionPlanVisitor visitor = new ExecutionPlanVisitor(filepath);
            o.visit(visitor);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    public long time_query_plan(String query) {
        StopWatch watch = new StopWatch();
        watch.start();
        Query q = QueryFactory.create(query);
        Op o = Algebra.compile(q);
        o = Algebra.optimize(o);
        // SSE.write(o);
        ExecutionPlanVisitor visitor = new ExecutionPlanVisitor(false);
        o.visit(visitor);
        watch.stop();
        long time = watch.getTime(TimeUnit.NANOSECONDS);
        return time;
    }

    // Times the query plan generation for each
    public void time_query_plan_extraction(String path, String outputFile) throws IOException {
        LSQreader reader = new LSQreader(path);
        DecimalFormat df = new DecimalFormat("##########0.00000000000000000000");
        FileWriter fw = new FileWriter(outputFile);
        CSVWriter c;
        c = new CSVWriter(fw);
        c.writeNext(new String[] { "id", "time" });

        java.util.ArrayList<String> data = reader.getNext();
        while (data != null) {
            String id = data.get(0);
            String query = data.get(1);
            try {
                System.out.println("Extracting for " + id);
                double time = time_query_plan(query);
                if (sub_id) {

                    // System.out.println(id.substring(20) + " " + String.valueOf());
                    c.writeNext(new String[] { id.substring(20), df.format(time / 1.0E9) });

                } else {
                    c.writeNext(new String[] { id, df.format(time / 1.0E9) });
                    // System.out.println(id + " " + String.valueOf(df.format(time / 1.0E9)));
                }
            } catch (org.apache.jena.query.QueryException e) {
                System.out.println("Did not work for " + id);
            }
            data = reader.getNext();
        }
        c.close();
        fw.close();
    }
    /*
     * The extracted queries are written to files with
     */
    public void extract_query_plan(LSQreader reader, String outputdir) {
        java.util.ArrayList<String> data = reader.getNext();
        while(data != null){
            String id = data.get(0);
            String query = data.get(1);
            try{
                System.out.println("Extracting for "+id);
            if(sub_id){
                	extract_query_plan(query, outputdir + "/" + id.substring(20));
            }else{
                        extract_query_plan(query, outputdir + "/" + String.valueOf(id));
            }}catch (org.apache.jena.query.QueryException e) {
                System.out.println("Did not work for " + id);
            }
            data = reader.getNext();
        }
        /*
        LinkedList<String> ids = reader.getIds();
        LinkedList<String> queries = reader.getQueries();
        for (int i = 0; i < ids.size(); i++) {
            try {
		if(sub_id){
                	extract_query_plan(queries.get(i), outputdir + "/" + ids.get(i).substring(20));
		}else{
                	extract_query_plan(queries.get(i), outputdir + "/" + String.valueOf(ids.get(i)));
		}
		} catch (org.apache.jena.query.QueryException e) {
                System.out.println("Did not work for " + ids.get(i));
            }
        } */
    }

}
