package semanticweb.sparql.preprocess;

import org.apache.jena.datatypes.xsd.impl.XSDBaseNumericType;
import org.apache.jena.graph.Node;
import org.apache.jena.query.Query;
import org.apache.jena.query.QueryFactory;
import org.apache.jena.sparql.algebra.Algebra;
import org.apache.jena.sparql.algebra.Op;
import org.apache.jena.sparql.core.TriplePath;
import org.apache.jena.sparql.syntax.*;
import semanticweb.sparql.Operator;
import java.util.*;

/**
 * Process a SPARQL query string and extract sets of tables, joins,predicates, and uriPredicates
 * for future use in DeepSet architecture.
 */
public class QueryFeatureExtractor {

    private ArrayList<String> queryTables;
    private ArrayList<String> queryVariables;
    private ArrayList<String> queryJoins;
    private HashMap<String, HashMap<String, ArrayList<String>>> queryJoinsV1Nodes;
    private ArrayList<HashMap<String, Object>> queryPredicates;
    private ArrayList<HashMap<String, Object>> queryPredicatesUris;
    private String query;
    private String id;
    private String cardinality;
    public static int TWO_INCOMING_PREDICATES = 1;
    public static int TWO_OUTGOING_PREDS = 2;
    public static int ONE_WAY_TWO_PREDS = 3;
    public static int TWO_OUTGOING_PRED_VAR_URI = 4;
    public static int TWO_INCOMING_PREDS_VAR_LITERAL = 5;
    /**
     * Constructor for query String
     * @param query
     */
    public QueryFeatureExtractor(String query) {
        this.query = query;
        this.queryTables = new ArrayList<>();
        this.queryVariables = new ArrayList<>();
        this.queryJoins = new ArrayList<>();
        this.queryJoinsV1Nodes = new HashMap<>();
        this.queryPredicates = new ArrayList<>();
        this.queryPredicatesUris = new ArrayList<>();
    }

    /**
     * Constructor for query data in a List
     * @param queryArr {@link ArrayList} [id, query, cardinality]
     */
    public QueryFeatureExtractor(ArrayList<String> queryArr) {
        this.id = queryArr.get(0);
        this.query = queryArr.get(1);
        this.cardinality = queryArr.get(2);
        this.queryTables = new ArrayList<>();
        this.queryVariables = new ArrayList<>();
        this.queryJoins = new ArrayList<>();
        this.queryJoinsV1Nodes = new HashMap<>();
        this.queryPredicates = new ArrayList<>();
        this.queryPredicatesUris = new ArrayList<>();
    }
    private void processTPFJoinsV1(Node subject, Node predicate, Node object){
        String subjectName = subject.getName();
        String objectName = object.getName();
        if (queryJoinsV1Nodes.containsKey(subjectName)){
            //If nodes contain the variable...
            HashMap<String, ArrayList<String>> node = queryJoinsV1Nodes.get(subjectName);
            // Register outgoing edge.
            node.get("outgoing").add(predicate.getURI());
        }
        else {
            HashMap<String, ArrayList<String>> node = new HashMap<>();
            ArrayList<String> list = new ArrayList<>();
            list.add(predicate.getURI());
            node.put("incoming", new ArrayList<>());
            node.put("outgoing_uri", new ArrayList<>());
            node.put("outgoing", list);
            queryJoinsV1Nodes.put(subjectName, node);

        }
        //Adding incoming edges of triple;
        if (queryJoinsV1Nodes.containsKey(objectName)){
            //If nodes contain the variable...
            HashMap<String, ArrayList<String>> node = queryJoinsV1Nodes.get(objectName);
            // Register outgoing edge.
            node.get("incoming").add(predicate.getURI());
        }
        else {
            HashMap<String, ArrayList<String>> node = new HashMap<>();
            ArrayList<String> list = new ArrayList<>();
            list.add(predicate.getURI());
            node.put("incoming", list);
            node.put("outgoing", new ArrayList<>());
            node.put("outgoing_uri", new ArrayList<>());
            queryJoinsV1Nodes.put(objectName,node);
        }
    }

    /**
     * Processing Join between Predicates to target a var and a Literal
     * @param subject
     * @param predicate
     * @param object
     */
    private void processTPFJoinsVar_URI(Node subject, Node predicate, Node object){
        String subjectName = subject.getName();
        if (queryJoinsV1Nodes.containsKey(subjectName)){
            //If nodes contain the variable...
            HashMap<String, ArrayList<String>> node = queryJoinsV1Nodes.get(subjectName);
            // Register outgoing edge that target to a literal Var -> pred -> uri
            node.get("outgoing_uri").add(predicate.getURI());
        }
        else {
            HashMap<String, ArrayList<String>> node = new HashMap<>();
            ArrayList<String> list = new ArrayList<>();
            list.add(predicate.getURI());
            node.put("outgoing", new ArrayList<>());
            node.put("incoming", new ArrayList<>());
            node.put("outgoing_uri", list);
            queryJoinsV1Nodes.put(subjectName, node);
        }
    }
    private void processVarPredVar(Node subject, Node predicate, Node object) {
        if (!this.queryTables.contains(predicate.getURI())) {
            this.queryTables.add(predicate.getURI());
        }
        //add variables subject to list
        if (!this.queryVariables.contains(subject.getName())) {
            this.queryVariables.add(subject.getName());
        }
        //add variables object to list
        if (!this.queryVariables.contains(object.getName())) {
            this.queryVariables.add(object.getName());
        }
        //add joins  var1_predicateURI_var2
        this.queryJoins.add(
                ""
                        .concat("v")
                        .concat(
                                String.valueOf(this.queryVariables.indexOf(subject.getName()))
                        )
                        .concat("_")
                        .concat(predicate.getURI())
                        .concat("_")
                        .concat("v")
                        .concat(
                                String.valueOf(this.queryVariables.indexOf(object.getName()))
                        )
        );
        this.processTPFJoinsV1(subject, predicate, object);
    }

    /**
     * Logic for subject var, predicate uri, object String literal like (Var1.foaf:name, 'daniel' )
     *
     * @param subject   Subject of the triple pattern frame
     * @param predicate Predicate  of the triple pattern frame
     * @param object    Object  of the triple pattern frame
     */
    private void processVarPredLiteral(Node subject, Node predicate, Node object) {
        if (!queryTables.contains(predicate.getURI())) {
            queryTables.add(predicate.getURI());
        }
        //add variables subject to list
        if (!queryVariables.contains(subject.getName())) {
            queryVariables.add(subject.getName());
        }
        //add Literal object to list
        HashMap<String, Object> pred = new HashMap<>();
        pred.put("col", predicate.getURI());
        pred.put("operator", Operator.EQUAL);
        pred.put("value", object.getLiteralValue());
        queryPredicates.add(pred);
    }

    /**
     * Logic for subject var, predicate uri, object int literal like (Var1.foaf:age, 29 )
     *
     * @param subject   Subject of the triple pattern frame
     * @param predicate Predicate  of the triple pattern frame
     * @param object    Object  of the triple pattern frame
     */
    private void processVarPredNumeric(Node subject, Node predicate, Node object) {
        if (!queryTables.contains(predicate.getURI())) {
            queryTables.add(predicate.getURI());
        }
        //add variables subject to list
        if (!queryVariables.contains(subject.getName())) {
            queryVariables.add(subject.getName());
        }
        //add Literal object to list
        HashMap<String, Object> pred = new HashMap<>();
        pred.put("col", predicate.getURI());
        pred.put("operator", Operator.EQUAL);
        pred.put("value", object.getLiteralValue());
        queryPredicates.add(pred);
    }

    /**
     * Logic for subject literal, predicate uri, object var literal like (29  foaf:age, var )
     *
     * @param subject   Subject of the triple pattern frame
     * @param predicate Predicate  of the triple pattern frame
     * @param object    Object  of the triple pattern frame
     */
    private void processNumericPredVar(Node subject, Node predicate, Node object) {
        if (!queryTables.contains(predicate.getURI())) {
            queryTables.add(predicate.getURI());
        }
        //add variables subject to list
        if (!queryVariables.contains(object.getName())) {
            queryVariables.add(object.getName());
        }
        //add Literal object to list
        HashMap<String, Object> pred = new HashMap<>();
        pred.put("col", predicate.getURI());
        pred.put("operator", Operator.EQUAL);
        pred.put("value", subject.getLiteralValue());
        queryPredicates.add(pred);
    }
    private String getQueryPredUri(String triple){
        String head = "SELECT (count( ?var )  AS ?total) WHERE { \n";
        String end = "\n }";
        return head.concat(triple).concat(end);
    }
    /**
     * Logic for subject var, predicate uri, object int literal like (Var1.rdf:type, foaf:Person )
     *
     * @param subject   Subject of the triple pattern frame
     * @param predicate Predicate  of the triple pattern frame
     * @param object    Object  of the triple pattern frame
     */
    private void processVarPredUri(Node subject, Node predicate, Node object) {
        if (!this.queryTables.contains(predicate.getURI())) {
            this.queryTables.add(predicate.getURI());
        }
        if (!this.queryVariables.contains(subject.getName())) {
            this.queryVariables.add(subject.getName());
        }
        HashMap<String, Object> pred = new HashMap<>();
        pred.put("col", predicate.getURI());
        pred.put("operator", Operator.EQUAL);
        pred.put("object", object.getURI());
        String hash = String.valueOf(pred.get("col")).concat(String.valueOf(pred.get("operator"))).concat(String.valueOf(pred.get("object")));
        pred.put("sampling_query_id", hash);

        pred.put("sampling_query", getQueryPredUri(
                "?var "
                        .concat("<")
                        .concat(String.valueOf(pred.get("col")))
                        .concat(">")
                        .concat(" ")
                        .concat("<")
                        .concat(String.valueOf(pred.get("object")))
                        .concat(">")
                )
        );
        this.queryPredicatesUris.add(pred);
        this.processTPFJoinsVar_URI(subject, predicate, object);
    }

    /**
     * Logic for subject uri, predicate uri, object int var like (:Boat :builtBy ?var1 )
     *
     * @param subject   Subject of the triple pattern frame
     * @param predicate Predicate  of the triple pattern frame
     * @param object    Object  of the triple pattern frame
     */
    private void processUriPredVar(Node subject, Node predicate, Node object) {
        if (!this.queryTables.contains(predicate.getURI())) {
            this.queryTables.add(predicate.getURI());
        }
        //add variables object to list
        if (!this.queryVariables.contains(object.getName())) {
            this.queryVariables.add(object.getName());
        }
        HashMap<String, Object> pred = new HashMap<>();
        pred.put("col", predicate.getURI());
        pred.put("operator", Operator.EQUAL);
        pred.put("object", subject.getURI());

        String hash = String.valueOf(pred.get("col")).concat(String.valueOf(pred.get("operator"))).concat(String.valueOf(pred.get("object")));
        pred.put("sampling_query_id", hash);

        pred.put("sampling_query", getQueryPredUri(
                "<"
                        .concat(String.valueOf(pred.get("object")))
                        .concat(">")
                        .concat(" ")
                        .concat("<")
                        .concat(String.valueOf(pred.get("col")))
                        .concat(">")
                        .concat(" ?var")
                        .concat( " . ")
        ));
        this.queryPredicatesUris.add(pred);
    }

    /**
     * Retrieve Map of query processed. Each Map contains:
     *  "queryTables"
     *  "queryVariables"
     *  "queryJoins"
     *  "queryPredicates"
     *  "queryPredicatesUris"
     * @return HashMap with the mentioned above keys.
     */
    public Map<String, Object> getProcessedData() {
        Query query = QueryFactory.create(this.query);

        // Generate algebra
        Op op = Algebra.compile(query);
        op = Algebra.optimize(op);

        Element e = query.getQueryPattern();
        HashMap<String, TriplePath> tpf = new HashMap<>();
        // AlgebraFeatureExtractor.getFeaturesDeepSet(op);
        // This will walk through all parts of the query
        ElementWalker.walk(e,
                // For each element...
                new ElementVisitorBase() {
                    // Delete tpf of Optional of queries from list to tpfs to vectorize.
                    public void visit(ElementOptional el) {
                        List<Element> elements = ((ElementGroup) el.getOptionalElement()).getElements();
                        for (Element element : elements) {
                            String key = element.toString();
                            tpf.remove(key);
                        }
                    }

                    // ...when it's a block of triples...
                    public void visit(ElementPathBlock el) {
                        // ...go through all the triples...
                        Iterator<TriplePath> triples = el.patternElts();
                        while (triples.hasNext()) {
                            TriplePath t = triples.next();
                            String tripleHash = t.getPath() != null ? t.getSubject() + " " + t.getPath() + " " + t.getObject() : t.getSubject() + " " + t.getPredicate() + " " + t.getObject();
                            tpf.put(tripleHash, t);
                        }
                    }
                }
        );
        for (TriplePath t : tpf.values()) {
            // Loop over elements without optional triples.
            // ...and grab the subject
            Node subject = t.getSubject();
            Node object = t.getObject();
            Node predicate = t.getPredicate();

            //           VAR                       URI                 VAR
            if (subject.isVariable() && predicate.isURI() && object.isVariable()) {
                //if not int table list add to.
                this.processVarPredVar(subject, predicate, object);
            }
            //           VAR                       URI                 URI
            else if (subject.isVariable() && predicate.isURI() && object.isURI()) {
                this.processVarPredUri(subject, predicate, object);
            }
            //           VAR                       URI                 LIT_NUMBER
            else if (
                    subject.isVariable() &&
                    predicate.isURI() &&
                    object.isLiteral() && object.getLiteralDatatype() != null && object.getLiteralDatatype().getClass() == XSDBaseNumericType.class) {
                this.processVarPredNumeric(subject, predicate, object);
            }
            //           VAR                       URI                 LITERAL
            else if (
                    subject.isVariable() &&
                    predicate.isURI() &&
                    object.isLiteral()) {
                    this.processVarPredLiteral(subject, predicate, object);
            }

            //Less probables
            //           URI                       URI                 VAR
            else if (subject.isURI() && predicate.isURI() && object.isVariable()) {
                this.processUriPredVar(subject, predicate, object);
            }
            //           LIT_Numeric               URI                 VAR
            else if (
                    subject.isLiteral() &&
                    subject.getLiteralDatatype() != null &&
                    subject.getLiteralDatatype().getClass() == XSDBaseNumericType.class && object.isVariable() && predicate.isURI()) {
                this.processNumericPredVar(subject, predicate, object);
            }
            // Todo Incorporate other cases...
        }
        ArrayList<ArrayList> queryJoinsV1Vec = getRepresentationJoins(this.queryJoinsV1Nodes);
        Map<String, Object> result = new HashMap<>();
        result.put("queryTables", this.queryTables);
        result.put("queryVariables", this.queryVariables);
        result.put("queryJoins", this.queryJoins);
        result.put("queryPredicates", this.queryPredicates);
        result.put("queryPredicatesUris", this.queryPredicatesUris);
        result.put("queryJoinsV1Vec", queryJoinsV1Vec);
        if (this.id != null) {
            result.put("id", this.id);
        }
        if (this.cardinality != null) {
            result.put("cardinality", this.cardinality);
        }
        return result;
    }

    public ArrayList<ArrayList> getRepresentationJoins(HashMap<String, HashMap<String, ArrayList<String>>> queryJoinsV1Nodes) {
        ArrayList<ArrayList> joisRep = new ArrayList<>();
        ArrayList<String> keys =  new ArrayList<>(queryJoinsV1Nodes.keySet());
        for (int i = 0; i <keys.size() ; i++) {
            HashMap<String, ArrayList<String>> stringArrayListHashMap = queryJoinsV1Nodes.get(keys.get(i));
            // Processing incomings...
            for (int j = 0; j < stringArrayListHashMap.get("incoming").size(); j++) {
                // Incoming -> node -> outgoings
                for (int k = 0; k < stringArrayListHashMap.get("outgoing").size(); k++) {
                    ArrayList<String> trainSample = new ArrayList<>();
                    trainSample.add(stringArrayListHashMap.get("incoming").get(j));
                    trainSample.add(stringArrayListHashMap.get("outgoing").get(k));
                    trainSample.add(String.valueOf(QueryFeatureExtractor.ONE_WAY_TWO_PREDS));
                    joisRep.add(trainSample);
                }
                // Incoming --> node <-- Incoming
                for (int k = 0; k < stringArrayListHashMap.get("incoming").size(); k++) {
                    String predA = stringArrayListHashMap.get("incoming").get(j);
                    String predB = stringArrayListHashMap.get("incoming").get(k);
                    //If are diferents then add to list
                    if(!predA.equals(predB)){
                        ArrayList<String> trainSample = new ArrayList<>();
                        trainSample.add(predA);
                        trainSample.add(predB);
                        trainSample.add(String.valueOf(QueryFeatureExtractor.TWO_INCOMING_PREDICATES));
                        joisRep.add(trainSample);
                    }
                }
            }
            // Processing outgoings...
            for (int j = 0; j < stringArrayListHashMap.get("outgoing").size(); j++) {
                // VAR <--OUTGOING <-- node --> OUTGOING --> VAR
                String predA = stringArrayListHashMap.get("outgoing").get(j);
                for (int k = 0; k < stringArrayListHashMap.get("outgoing").size(); k++) {
                    String predB = stringArrayListHashMap.get("outgoing").get(k);
                    //If are diferents then add to list
                    if(!predA.equals(predB)){
                        ArrayList<String> trainSample = new ArrayList<>();
                        trainSample.add(predA);
                        trainSample.add(predB);
                        trainSample.add(String.valueOf(QueryFeatureExtractor.TWO_OUTGOING_PREDS));
                        joisRep.add(trainSample);
                    }
                }
                // VAR <--OUTGOING <-- node --> OUTGOING --> URI
                for (int k = 0; k < stringArrayListHashMap.get("outgoing_uri").size(); k++) {
                    String predB = stringArrayListHashMap.get("outgoing_uri").get(k);
                    //If are diferents then add to list
                    if(!predA.equals(predB)){
                        ArrayList<String> trainSample = new ArrayList<>();
                        trainSample.add(predA);
                        trainSample.add(predB);
                        trainSample.add(String.valueOf(QueryFeatureExtractor.TWO_OUTGOING_PRED_VAR_URI));
                        joisRep.add(trainSample);
                    }
                }
            }
        }
        return joisRep;
    }
    public static void main(String[] args) {
        String s = "PREFIX foaf: <http://xmlns.com/foaf/0.1/> SELECT ?name ?email WHERE {  ?x foaf:knows ?y . ?y foaf:name ?name .  OPTIONAL { ?y foaf:mbox ?email }  }";
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
                "SELECT DISTINCT ?uri ?string \n" +
                "WHERE {\n" +
                "\t?uri rdf:type dbo:Book .\n" +
                "        ?uri dbo:author res:Danielle_Steel .\n" +
                "        ?uri  <http://dbpedia.org/ontology/numberOfPages>  336 . \n" +
                "\tOPTIONAL { ?uri rdfs:label ?string . FILTER (lang(?string) = 'en') }\n" +
                "}";
        String[] queries;
        queries = new String[4];
        queries[0] = s;
        queries[1] = s1;
        queries[2] = s2;
        queries[3] = s3;
        for (String query : queries) {
            QueryFeatureExtractor qfe = new QueryFeatureExtractor(query);
            Map<String, Object> data = qfe.getProcessedData();
            System.out.println(data);
        }
    }
}
