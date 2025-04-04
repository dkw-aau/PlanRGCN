package semanticweb.sparql.preprocess;


import org.apache.jena.graph.Node;
import org.apache.jena.query.Query;
import org.apache.jena.query.QueryFactory;
import org.apache.jena.sparql.algebra.Algebra;
import org.apache.jena.sparql.algebra.Op;
import org.apache.jena.sparql.core.TriplePath;
import org.apache.jena.sparql.syntax.*;

import java.util.*;

public class RecurrentQueryFeatureExtractor {
    protected ArrayList<String> queryPredicates;
    protected ArrayList<HashMap<String, String>> querytpfs;
    protected String query;
    protected String id;
    protected String execTime;

    public static String VAR_PRED_VAR = "1";
    public static String VAR_PRED_LIT = "2";
    public static String LIT_PRED_VAR = "3";
    public static String VAR_PRED_URI = "4";
    public static String LIT_VAR_URI = "5";
    public static String URI_VAR_LIT = "6";
    public static String URI_VAR_URI = "7";
    public static String LIT_VAR_LIT = "8";
    public static String URI_PRED_VAR = "9";

    /**
     * Constructor for query data in a List
     * @param queryArr {@link ArrayList} [id, query, cardinality]
     */
    public RecurrentQueryFeatureExtractor(ArrayList<String> queryArr) {
        this.id = queryArr.get(0);
        this.query = queryArr.get(1);
        this.execTime = queryArr.get(2);
        this.querytpfs = new ArrayList<>();
        this.queryPredicates = new ArrayList<>();
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
            if (predicate.isURI()) {
                if (subject.isVariable()  && object.isVariable()) {
                    //if not int table list add to.
                    this.processTpfWithPreds(subject, predicate, object, VAR_PRED_VAR);
                }
                else if (subject.isVariable()  && object.isURI()) {
                    this.processTpfWithPreds(subject, predicate, object, VAR_PRED_URI);
                }
                else if (subject.isVariable()  && object.isLiteral()) {
                    this.processTpfWithPreds(subject, predicate, object, VAR_PRED_LIT);
                }
                //Less probables
                else if (subject.isURI()  && object.isVariable()) {
                    this.processTpfWithPreds(subject, predicate, object, URI_PRED_VAR);
                }
                else if (subject.isLiteral()  && object.isVariable()) {
                    this.processTpfWithPreds(subject, predicate, object, LIT_PRED_VAR);
                }
            }
            else if(predicate.isVariable()) {
                if (subject.isLiteral() && object.isLiteral()) {
                    this.processNoPreds(subject, predicate, object, LIT_VAR_LIT);
                }
                else if (
                        subject.isLiteral() && object.isURI()) {
                    this.processNoPreds(subject, predicate, object, LIT_VAR_URI);
                }
                else if (
                        subject.isURI() && object.isLiteral()) {
                    this.processNoPreds(subject, predicate, object, URI_VAR_LIT);
                }
                else if (
                        subject.isURI() && object.isURI()) {
                    this.processNoPreds(subject, predicate, object, URI_VAR_URI);
                }
            }
            // Todo Incorporate other cases...
        }
        Map<String, Object> result = new HashMap<>();
        result.put("queryPredicates", this.queryPredicates);
        result.put("tpfs", this.querytpfs);
        if (this.id != null) {
            result.put("id", this.id);
        }
        if (this.execTime != null) {
            result.put("execTime", this.execTime);
        }
        return result;
    }

    void processTpfWithPreds(Node subject, Node predicate, Node object, String structure) {
        if(!this.queryPredicates.contains(predicate.getURI())) {
            this.queryPredicates.add(predicate.getURI());
        }
        HashMap<String, String> pred = new HashMap<>();
        pred.put("predicate", predicate.getURI());
        pred.put("structure", structure);
        querytpfs.add(pred);
    }
    void processNoPreds(Node subject, Node predicate, Node object, String structure) {

        HashMap<String, String> pred = new HashMap<>();
        pred.put("predicate", "NONE");
        pred.put("structure", structure);
        querytpfs.add(pred);
    }
}
