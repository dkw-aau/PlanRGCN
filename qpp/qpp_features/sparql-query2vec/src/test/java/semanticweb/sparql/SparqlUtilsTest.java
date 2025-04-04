package semanticweb.sparql;

import org.apache.jena.graph.Triple;
import org.apache.jena.rdf.model.Model;
import org.junit.jupiter.api.Test;
import util.Graph;

import java.util.LinkedList;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class SparqlUtilsTest {
    /*
    * The goal of the retrieveTriple unit test is to verify which operations are supported in the SPARQL Algebra Statististics.
    * Conclusions so far.
    *   - triples in FILTER EXISTS/NOT EXISTS clauses are not extracted, meaning only partial operations are considered for these queries.
    *   - Triples in optional are considered as part of the set of "regular" triples-
    *   - Triples in unions are extracted and used for the building of the query graph.
    *   - Property path seems to work as well but the representation does not seem to make sense because the representation
    *  of property path predicate is the string extraction of the entire operation.
    * */
    @Test
    void retrieveTriples01() {
        String query = """
                SELECT * {
                ?s ?p ?o.
                OPTIONAL {
                ?s <http://rfdtype> <http://exmaple.org#Person>
                
                }
                }
                """;
        Set<Triple> triples = SparqlUtils.retrieveTriples(query);
        for (Triple i: triples
             ) {
            System.out.println(i);
        }

    }
    @Test
    void retrieveTriples02() {
        String query = """
                SELECT * {
                ?s ?p ?o.
                FILTER NOT EXISTS {
                ?s <http://rfdtype> <http://exmaple.org#Person>
                
                }
                }
                """;
        Set<Triple> triples = SparqlUtils.retrieveTriples(query);
        LinkedList<String> objects = new LinkedList<>();
        for (Triple i: triples
        ) {
            if (i.getObject().isURI()){
                objects.add(i.getObject().getURI());
            }
        }
        assertTrue(!objects.contains("http://exmaple.org#Person"));

    }

    @Test
    void retrieveTriples03() {
        String query = """
                SELECT * {{
                ?s ?p ?o.
                }
                UNION {
                ?s <http://rfdtype> <http://exmaple.org#Person>
                
                }
                }
                """;
        Set<Triple> triples = SparqlUtils.retrieveTriples(query);
        LinkedList<String> objects = new LinkedList<>();
        for (Triple i: triples
        ) {
            if (i.getObject().isURI()){
                objects.add(i.getObject().getURI());
            }
        }
        assertTrue(objects.contains("http://exmaple.org#Person"));

    }

    @Test
    void retrieveTriples04() {
        String query = """
                SELECT * {
                ?s (<http://rfdtype>|^<http://rfdtype>)* <http://exmaple.org#Person>
                }
                """;
        Set<Triple> triples = SparqlUtils.retrieveTriples(query);
        LinkedList<String> objects = new LinkedList<>();
        for (Triple i: triples
        ) {
            if (i.getObject().isURI()){
                objects.add(i.getObject().getURI());
            }
        }
        assertTrue(objects.contains("http://exmaple.org#Person"));

    }

    @Test
    void buildQueryRDFGraph01() {
        String query = """
                SELECT * {
                ?s <http://rfdtype> <http://exmaple.org#Person>
                }
                """;
        Model model = SparqlUtils.buildQueryRDFGraph(query);
        System.out.println(model);

    }

    @Test
    void buildSPARQL2GXLGraph01() {
        String query = """
                SELECT * {
                ?s <http://rfdtype> <http://exmaple.org#Person> .
                ?s ?p ?o
                }
                """;

        try {
            Graph query1 = SparqlUtils.buildSPARQL2GXLGraph(query, "query1");
            System.out.println(query1);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }
}