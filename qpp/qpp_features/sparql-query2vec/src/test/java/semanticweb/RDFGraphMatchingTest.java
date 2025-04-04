package semanticweb;

import org.junit.jupiter.api.Test;
import semanticweb.sparql.SparqlUtils;
import util.Graph;

import java.sql.SQLOutput;

import static org.junit.jupiter.api.Assertions.*;

/*
* This class will test and provide insights into how the graph edit distance is computed for the baselines*/
class RDFGraphMatchingTest {
    String getQuery1() {
        return """
                SELECT * {
                ?s <http://rfdtype> <http://exmaple.org#Person> .
                ?s ?p ?o
                }
                """;
    }
    String getQuery1_diff_var(){
        return """
                SELECT * {
                ?s1 <http://rfdtype> <http://exmaple.org#Person> .
                ?s1 ?p ?o
                }
                """;
    }
    String getQuery2(){
        return """
                SELECT * {
                ?s <http://rfdtype> <http://exmaple.org#Person1> .
                ?s ?p <http://exmaple.org#Person2>.
                ?s ?p <http://exmaple.org#Person3>.
                ?s ?p <http://exmaple.org#Person4>.
                ?s ?p <http://exmaple.org#Person4>.
                }
                """;
    }

    @Test
    void distanceBipartiteHungarian() {
        String query = getQuery1();

        try {
            Graph query1 = SparqlUtils.buildSPARQL2GXLGraph(query, "query1");
            Graph query2 = SparqlUtils.buildSPARQL2GXLGraph(query, "query2");
            RDFGraphMatching rdfGraphMatching = new RDFGraphMatching();
            double val = rdfGraphMatching.distanceBipartiteHungarian(query1,query2);
            System.out.println(val);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
    @Test
    void distanceBipartiteHungarian02() {
        /*
        * This Graph edit distance is not exact but an approximation, making it hard to use the exact graph edit distances as a measure of this.*/
        String query = getQuery1();
        String query2 = getQuery2();
        try {
            Graph graph1 = SparqlUtils.buildSPARQL2GXLGraph(query, "query1");
            Graph graph2 = SparqlUtils.buildSPARQL2GXLGraph(query2, "query2");
            RDFGraphMatching rdfGraphMatching = new RDFGraphMatching();
            double val = rdfGraphMatching.distanceBipartiteHungarian(graph1,graph2);
            System.out.println(val);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}