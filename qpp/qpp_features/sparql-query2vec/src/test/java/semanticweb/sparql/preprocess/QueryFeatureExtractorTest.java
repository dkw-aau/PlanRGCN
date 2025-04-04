package semanticweb.sparql.preprocess;
import org.apache.jena.query.ResultSet;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import semanticweb.sparql.SparqlUtils;
import semanticweb.sparql.utils.DBPediaUtils;
import static org.junit.jupiter.api.Assertions.assertEquals;
import java.util.*;
@Disabled
public class QueryFeatureExtractorTest {

    @Disabled
    @Test
    public void testpProcessTPFJoinsV1() {

        String s = "PREFIX foaf:    <http://xmlns.com/foaf/0.1/> SELECT ?name ?email WHERE {  ?x foaf:knows ?y . ?y foaf:name ?name .  OPTIONAL { ?y foaf:mbox ?email }  }";
        String s1 = "PREFIX dbo: <http://dbpedia.org/ontology/>\n" +
                "PREFIX res: <http://dbpedia.org/resource/>\n" +
                "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n" +
                "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n" +
                "SELECT DISTINCT ?uri ?other \n" +
                "WHERE {\n" +
                "\t?uri rdf:type dbo:Film .\n" +
                "        ?uri dbo:starring res:Julia_Roberts .\n" +
                "        ?uri dbo:starring ?other.\n" +
                "}";
        String s2 = "PREFIX dbo: <http://dbpedia.org/ontology/>\n" +
                "PREFIX res:  <http://dbpedia.org/resource/>\n" +
                "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n" +
                "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n" +
                "SELECT DISTINCT ?uri ?string \n" +
                "WHERE {\n" +
                "\t?uri rdf:type dbo:Book .\n" +
                "        ?uri dbo:author res:Danielle_Steel .\n" +
                "\tOPTIONAL { ?uri rdfs:label ?string . FILTER (lang(?string) = 'en') }\n" +
                "}";
        String s3 = "PREFIX dbo: <http://dbpedia.org/ontology/>\n" +
                "PREFIX res:  <http://dbpedia.org/resource/>\n" +
                "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n" +
                "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n" +
                "SELECT DISTINCT ?uri ?string ?wife \n" +
                "WHERE {\n" +
                "\t?uri rdf:type dbo:Book .\n" +
                "        ?uri dbo:author ?author .\n" +
                "        ?wife dbo:wifeOf ?author .\n" +
                "        ?uri  <http://dbpedia.org/ontology/numberOfPages>  336 . \n" +
                "\tOPTIONAL { ?uri rdfs:label ?string . FILTER (lang(?string) = 'en') }\n" +
                "}";
        //The numbers meaning id , query, cardinality, irrelevant for this test.
        ArrayList<String> queryS0 = new ArrayList<>(Arrays.asList("1", s, "212"));
        ArrayList<String> queryS1 = new ArrayList<>(Arrays.asList("2", s1, "223"));
        ArrayList<String> queryS2 = new ArrayList<>(Arrays.asList("2", s2, "22"));
        ArrayList<String> queryS3 = new ArrayList<>(Arrays.asList("3", s3, "232"));
        ArrayList<ArrayList<String>> queries = new ArrayList<>(Arrays.asList(queryS0, queryS1, queryS2, queryS3));
        ArrayList<Object> result = new ArrayList<>();
        for (int i = 0; i < queries.size(); i++) {
            ArrayList<String> queryArr = queries.get(i);
            try {

                QueryFeatureExtractor qfe = new QueryFeatureExtractor(queryArr);
                Map<String, Object> queryVecData = qfe.getProcessedData();
                ArrayList<ArrayList> aList = (ArrayList<ArrayList>) queryVecData.get("queryJoinsV1Vec");
                result.add(aList);
                switch (i) {
                    case 0:
                        assertEquals  (aList.get(0).get(2), String.valueOf(QueryFeatureExtractor.ONE_WAY_TWO_PREDS));
                        break;
                    case 1:
                        assertEquals (aList.get(0).get(2), String.valueOf(QueryFeatureExtractor.TWO_OUTGOING_PRED_VAR_URI));
                        break;
                    case 2:
                        assertEquals (aList.size(),0);
                        break;
                    case 3:
                        assertEquals (aList.get(0).get(2), String.valueOf(QueryFeatureExtractor.TWO_INCOMING_PREDICATES));
                        break;
                }
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
        result.size();
    }
    @Disabled
    @Test
    public void testExecutionQuery() {

        String query = "PREFIX : <http://dbpedia.org/resource/>\n" +
                "SELECT (count( ?person )  AS ?total) WHERE {\n" +
                "      ?person dbo:birthPlace :Berlin .  \n" +
                "}";
        HashMap<String, String> namespases = SparqlUtils.getNamespacesStr("/home/daniel/Documentos/Web_Semantica/Work/Sparql2vec/prefixes.txt");
        int result = DBPediaUtils.execQueryCount(query, "https://dbpedia.org/sparql",  namespases);
        System.out.println(result);
    }
    @Disabled
    @Test
    public void testExecutionTripleQuery() {

        String query = "?var dbo:birthPlace :Berlin .";
        HashMap<String, String> namespases = SparqlUtils.getNamespacesStr("/home/daniel/Documentos/Web_Semantica/Work/Sparql2vec/prefixes.txt");
        int result = DBPediaUtils.execQueryCountTripleWithNS(query, "https://dbpedia.org/sparql",  namespases);
        System.out.println(result);
    }
}
