package semanticweb.sparql.preprocess;

import org.apache.jena.datatypes.xsd.impl.XSDBaseNumericType;
import org.apache.jena.graph.*;
import org.apache.jena.query.*;

import org.apache.jena.sparql.algebra.Algebra;
import org.apache.jena.sparql.algebra.Op;
import org.apache.jena.sparql.core.TriplePath;
import org.apache.jena.sparql.expr.E_IsNumeric;
import org.apache.jena.sparql.expr.Expr;
import org.apache.jena.sparql.expr.ExprFunction2;
import org.apache.jena.sparql.expr.ExprVar;
import org.apache.jena.sparql.expr.aggregate.AggCountVarDistinct;
import org.apache.jena.sparql.expr.aggregate.AggMax;
import org.apache.jena.sparql.expr.aggregate.AggMin;
import org.apache.jena.sparql.expr.nodevalue.NodeValueInteger;
import org.apache.jena.sparql.syntax.*;
import semanticweb.sparql.utils.PredHistogram;

import java.util.*;
public class RLQueryFeatureExtractor {
    private ArrayList<String> queryTables;
    private ArrayList<String> queryVariables;
    private HashMap<String, HashMap<String, ArrayList<String>>> queryJoinsNodes;
    private ArrayList<HashMap<String, Object>> queryPredicates;
    private ArrayList<HashMap<String, Object>> queryPredicatesUris;
    private String query;
    private String id;
    private String execTime;
    private String cardinality;
    private HashMap<String, PredHistogram> predQueryToHist;
    public static int TWO_INCOMING_PREDICATES = 1;
    public static int TWO_OUTGOING_PREDS = 2;
    public static int ONE_WAY_TWO_PREDS = 3;
    public static int TWO_OUTGOING_PRED_VAR_URI = 4;
    public static int TWO_INCOMING_PREDS_VAR_LITERAL = 5;
    public Map<String,Integer> Operator;
    public Map<Integer,String> OperatorInv;


    private HashMap<String, HashMap<String,Object>> filters;
    /**
     * Constructor for query String
     * @param query
     */
    public RLQueryFeatureExtractor(String query) {
        this.Operator = new HashMap<>();
        this.Operator.put("<" , -2);
        this.Operator.put("<=", -1);
        this.Operator.put("=" , 0);
        this.Operator.put(">=", 1);
        this.Operator.put(">",  2);
        this.OperatorInv.put(-2, "<");
        this.OperatorInv.put(-1, "<=");
        this.OperatorInv.put( 0, "=");
        this.OperatorInv.put( 1, ">=");
        this.OperatorInv.put( 2, ">");

        this.query = query;
        this.queryTables = new ArrayList<>();
        this.queryVariables = new ArrayList<>();
        this.queryJoinsNodes = new HashMap<>();
        this.queryPredicates = new ArrayList<>();
        this.queryPredicatesUris = new ArrayList<>();
        this.filters = new HashMap<>();
        this.predQueryToHist = new HashMap<>();
    }

    /**
     * Constructor for query data in a List
     * @param queryArr {@link ArrayList} [id, query, cardinality]
     */
    public RLQueryFeatureExtractor(String[] queryArr) {
        this.id = queryArr[0];
        this.query = queryArr[1];
        this.execTime = queryArr[2];
        this.queryTables = new ArrayList<>();
        this.queryVariables = new ArrayList<>();
        this.queryJoinsNodes = new HashMap<>();
        this.queryPredicates = new ArrayList<>();
        this.queryPredicatesUris = new ArrayList<>();
        this.predQueryToHist = new HashMap<>();

    }

    /**
     * Retrieve Map of query processed. Each Map contains:
     *  "queryTables"
     *  "queryVariables"
     *  "queryPredicates"
     *  "queryPredicatesUris"
     * @return HashMap with the mentioned above keys.
     */
    public Map<String, Object> getProcessedData() {
        Query query = null;
        try{
        query = QueryFactory.create(this.query);
        }catch(QueryParseException e){
            e.printStackTrace();
            System.out.println(this.query);
        }
        // Generate algebra
        Op op = Algebra.compile(query);
        op = Algebra.optimize(op);

        Element e = query.getQueryPattern();
        HashMap<String, TriplePath> tpf = new HashMap<>();
        HashMap<String, HashMap<String, Object>> filters = new HashMap<>();
        // AlgebraFeatureExtractor.getFeaturesDeepSet(op);
        // This will walk through all parts of the query
        ElementWalker.walk(e,
                // For each element...
                new ElementVisitorBase() {
                    public void visit(ElementFilter el) {
                        try {
                            ExprFunction2 exp = (ExprFunction2) el.getExpr();
                            Expr arg1 = exp.getArg1();
                            Expr arg2 = exp.getArg2();
                            HashMap<String,Object> expFilter = new HashMap<>();
                            if (arg1.isVariable() && ((NodeValueInteger) arg2).isNumber() ) {
                                expFilter.put("operator", exp.getOpName());
                                expFilter.put("value", ((NodeValueInteger) arg2).getDouble());
                                filters.put(arg1.getVarName(), expFilter);
                            }
                            else if(arg2.isVariable() && ((NodeValueInteger) arg1).isNumber()) {

                                String operator = exp.getOpName();
                                // invert operator
                                if(Operator.containsKey(operator)){
                                    operator = OperatorInv.get(Operator.get(operator) * -1);
                                }
                                expFilter.put("operator", operator);
                                expFilter.put("value", ((NodeValueInteger) arg1).getDouble());
                                filters.put(arg2.getVarName(), expFilter);
                            }
                        }
                        catch (Exception ex){
                            //Todo..
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
        this.filters = filters;
        for (TriplePath t : tpf.values()) {
            // Loop over elements without optional triples.
            // ...and grab the subject
            Node subject = t.getSubject();
            Node object = t.getObject();
            Node predicate = t.getPredicate();

            //           VAR                       URI                 VAR
            if(subject == null || predicate == null  || object == null){
                continue;
            }
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
        
        ArrayList<ArrayList> queryJoinsVec = getRepresentationJoinsLite(this.queryJoinsNodes);
        Map<String, Object> result = new HashMap<>();
        result.put("queryTables", this.queryTables);
        result.put("queryVariables", this.queryVariables);
        result.put("queryPredicates", this.queryPredicates);
        result.put("queryPredicatesUris", this.queryPredicatesUris);
        result.put("queryJoinsVec", queryJoinsVec);
        result.put("queriesToHist", this.predQueryToHist);
        if (this.id != null) {
            result.put("id", this.id);
        }
        if (this.cardinality != null) {
            result.put("cardinality", this.cardinality);
        }
        if (this.cardinality != null) {
            result.put("execTime", this.execTime);
        }
        return result;
    }

    /**
     * Procesa un Joins 
     * @param subject tpf
     * @param predicate tpf
     * @param object tpf
     */
    private void processTPFJoins(Node subject, Node predicate, Node object){
        String subjectName = subject.getName();
        String objectName = object.getName();
        if (queryJoinsNodes.containsKey(subjectName)){
            //If nodes contain the variable...
            HashMap<String, ArrayList<String>> node = queryJoinsNodes.get(subjectName);
            // Register outgoing edge.
            node.get("outgoing").add(predicate.getURI());
        }
        else {
            HashMap<String, ArrayList<String>> node = new HashMap<>();
            ArrayList<String> list = new ArrayList<>();
            list.add(predicate.getURI());
            node.put("incoming", new ArrayList<>());
            node.put("incoming_uri", new ArrayList<>());
            node.put("outgoing", list);
            node.put("outgoing_uri", new ArrayList<>());
            queryJoinsNodes.put(subjectName, node);
        }
        //Adding incoming edges of triple;
        if (queryJoinsNodes.containsKey(objectName)){
            //If nodes contain the variable...
            HashMap<String, ArrayList<String>> node = queryJoinsNodes.get(objectName);
            // Register incoming edge.
            node.get("incoming").add(predicate.getURI());
        }
        else {
            HashMap<String, ArrayList<String>> node = new HashMap<>();
            ArrayList<String> list = new ArrayList<>();
            list.add(predicate.getURI());
            node.put("incoming", list);
            node.put("incoming_uri", new ArrayList<>());
            node.put("outgoing", new ArrayList<>());
            node.put("outgoing_uri", new ArrayList<>());
            queryJoinsNodes.put(objectName,node);
        }
    }

    /**
     * Processing Join between Predicates to target a var and a Literal_0R_URI
     * @param subject
     * @param predicate
     * @param object
     */
    private void processTPFJoinsVar_URI(Node subject, Node predicate, Node object){
        String subjectName = subject.getName();
        if (queryJoinsNodes.containsKey(subjectName)) {
            //If nodes contain the variable...
            HashMap<String, ArrayList<String>> node = queryJoinsNodes.get(subjectName);
            // Register outgoing edge that target to a literal Var -> pred -> uri
            node.get("outgoing_uri").add(predicate.getURI());
        }
        else {
            HashMap<String, ArrayList<String>> node = new HashMap<>();
            ArrayList<String> list = new ArrayList<>();
            list.add(predicate.getURI());
            node.put("outgoing", new ArrayList<>());
            node.put("incoming", new ArrayList<>());
            node.put("incoming_uri", new ArrayList<>());
            node.put("outgoing_uri", list);
            queryJoinsNodes.put(subjectName, node);
        }
    }

    /**
     * Processing Join between Predicates to target a Literal_URI and a  VAR
     * @param subject
     * @param predicate
     * @param object
     */
    private void processTPFJoinsURI_Var(Node subject, Node predicate, Node object){
        String objectName = object.getName();
        if (queryJoinsNodes.containsKey(objectName)) {
            //If nodes contain the variable...
            HashMap<String, ArrayList<String>> node = queryJoinsNodes.get(objectName);
            // Register outgoing edge that target to a literal Var -> pred -> uri
            node.get("incoming_uri").add(predicate.getURI());
        }
        else {
            HashMap<String, ArrayList<String>> node = new HashMap<>();
            ArrayList<String> list = new ArrayList<>();
            list.add(predicate.getURI());
            node.put("outgoing", new ArrayList<>());
            node.put("incoming", new ArrayList<>());
            node.put("outgoing_uri", new ArrayList<>());
            node.put("incoming_uri", list);
            queryJoinsNodes.put(objectName, node);
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
        HashMap<String, Object> pred = this.createPredMap(subject, predicate, object, "ALL",true,true);
        this.processTPFJoins(subject, predicate, object);
        this.queryPredicates.add(pred);

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
        HashMap<String, Object> pred = this.createPredMap(subject, predicate, object, object.getLiteralValue(),true,false);
//        pred.put("col", predicate.getURI());
//        pred.put("operator", "=");
//        pred.put("value", object.getLiteralValue());
        //Todo change if needed and diferent method for literals
        this.processTPFJoinsVar_URI(subject, predicate, object);
        this.queryPredicates.add(pred);

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
        HashMap<String, Object> pred = this.createPredMap(subject, predicate, object, object.getLiteralValue(),true,false);
        String queryFilter = this.getQueryPredNumericByVar(
                new Node_Variable("sub"),
                predicate,
                new Node_Variable("obj"),
                "obj");
//        HashMap<String,String> qFilterDict = new HashMap<>();
//        qFilterDict.put("type","numeric");
//        qFilterDict.put("on","obj");
//        qFilterDict.put("query", queryFilter);
//        predQueryToHist.put(predicate.getURI(),qFilterDict);
        addQueryFilter("numeric","obj", queryFilter, predicate.getURI());

        queryPredicates.add(pred);
        //Todo change if needed and diferent method for literals
        this.processTPFJoinsVar_URI(subject, predicate, object);
    }

    /**
     *
     * @param subject
     * @param predicate
     * @param object
     * @param Initvalue
     * @param filterInSub
     * @param filterInObj
     * @return
     */
    private HashMap<String,Object> createPredMap(Node subject, Node predicate, Node object, Object Initvalue, boolean filterInSub, boolean filterInObj) {
        HashMap<String, Object> pred = new HashMap<>();
        pred.put("col", predicate.getURI());
        pred.put("operator", "=");
        pred.put("value", Initvalue);

        //Agregando key con el lugar donde se ejecuta el filtro para luego dividir los histogramas si son en el sujeto u objeto.
        if(filterInSub){
            pred.put("on",  "sub");
        }
        else if(filterInObj) {
            pred.put("on",  "obj");
        }

        // If variable in subject or object ar in query filters restrict values an change operator by operator and value inside filter.
        if (filterInSub && this.filters.containsKey(subject.getName())) {
            HashMap<String,Object> exprFilter = this.filters.get(subject.getName());
            String queryFilter = this.getQueryPredNumericByVar(new Node_Variable("sub"), predicate, new Node_Variable("obj"), "sub");
//            HashMap<String,String> qFilterDict = new HashMap<>();
//            qFilterDict.put("type","numeric");
//            qFilterDict.put("on","sub");
//            qFilterDict.put("query", queryFilter);
//            predQueryToHist.put(predicate.getURI(),qFilterDict);
            addQueryFilter("numeric","sub", queryFilter, predicate.getURI());

            pred.put("operator", exprFilter.get("operator"));
            pred.put("value", Integer.parseInt((String) exprFilter.get("value")));
            pred.put("on",  exprFilter.get("on"));
        }
        else if (filterInObj && this.filters.containsKey(object.getName()) ) {
            HashMap<String,Object> exprFilter = this.filters.get(object.getName());
            String queryFilter = this.getQueryPredNumericByVar(new Node_Variable("sub"), predicate, new Node_Variable("obj"), "obj");
//            HashMap<String,String> qFilterDict = new HashMap<>();
//            qFilterDict.put("type","numeric");
//            qFilterDict.put("on","obj");
//            qFilterDict.put("query", queryFilter);
//            predQueryToHist.put(predicate.getURI(), qFilterDict);
            addQueryFilter("numeric","obj", queryFilter, predicate.getURI());

            pred.put("operator", exprFilter.get("operator"));
            pred.put("value",  exprFilter.get("value"));
            pred.put("on",  exprFilter.get("on"));
        }
        return pred;
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
        HashMap<String, Object> pred =  this.createPredMap(subject, predicate, object, subject.getLiteralValue(),false,true);
        String queryFilter = this.getQueryPredNumericByVar(
                new Node_Variable("sub"),
                predicate,
                new Node_Variable("obj"),
                "sub");
//        HashMap<String,String> qFilterDict = new HashMap<>();
//        qFilterDict.put("type", "numeric");
//        qFilterDict.put("on", "sub");
//        qFilterDict.put("query", queryFilter);
//        predQueryToHist.put(predicate.getURI(),qFilterDict);
        addQueryFilter("numeric","sub", queryFilter, predicate.getURI());
        queryPredicates.add(pred);
    }
    private String getQueryPredUri(String triple){
        String head = "SELECT (count( ?var )  AS ?total) WHERE { \n";
        String end = "\n }";
        return head.concat(triple).concat(end);
    }
    private void addQueryFilter(String type, String on, String query, String predicate){
        if (predQueryToHist.containsKey(predicate)){
            PredHistogram predHist = predQueryToHist.get(predicate);
            predHist.setQuery(on, type, query);
            predQueryToHist.put(predicate, predHist);
        }
        else {
            PredHistogram predHist = new PredHistogram(type, on, query, predicate);
            predQueryToHist.put(predicate,predHist);
        }
    }
    /**
     * Create a String with query
     * @param s Node Subject
     * @param p Node Predicate
     * @param o Node Object
     * @param var
     * @return
     */
    private String getQueryPredUriByVar(Node s, Node p, Node o, String var){
        ElementTriplesBlock block = new ElementTriplesBlock(); // Make a BGP
        Triple pattern = Triple.create(s, p, o);
        block.addTriple(pattern);                              // Add our pattern match
        ElementGroup body = new ElementGroup();                // Group our pattern match and filter
        body.addElement(block);
        Query q = QueryFactory.make();
        q.setQueryPattern(body);                               // Set the body of the query to our group
        q.setQuerySelectType();                                // Make it a select query
        q.addResultVar(var);
        q.setLimit(1000);
        return q.toString();
    }
    /**
     * Create a String with query
     * @param s
     * @param p
     * @param o
     * @param var
     * @return
     */
    private String getQueryPredNumericByVar(Node s, Node p, Node o, String var){
        ElementTriplesBlock block = new ElementTriplesBlock(); // Make a BGP
        Triple pattern = Triple.create(s, p, o);
        block.addTriple(pattern);                              // Add our pattern match
        ElementGroup body = new ElementGroup();                // Group our pattern match and filter
        body.addElement(block);
        Query q = QueryFactory.make();
        //Adding filter para valores numericos
        Expr e = new E_IsNumeric(new ExprVar(var));
        ElementFilter filter = new ElementFilter(e);           // Make a filter matching the expression
        body.addElement(filter);

        q.setQueryPattern(body);                               // Set the body of the query to our group
        q.setQuerySelectType();                               // Make it a select query
        //Adding min max agregators over var
        AggCountVarDistinct count = new AggCountVarDistinct(new ExprVar(var));
        AggMax max = new AggMax(new ExprVar(var));
        AggMin min = new AggMin(new ExprVar(var));
        Expr maxExpr =  q.allocAggregate(max);
        Expr minExpr =  q.allocAggregate(min);
        Expr distExpr =  q.allocAggregate(count);
        q.addResultVar("max", maxExpr);
        q.addResultVar("min", minExpr);
        q.addResultVar("distinct", distExpr);
        q.setLimit(1000);
        return q.toString();
    }

    /**var
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
        HashMap<String, Object> pred = this.createPredMap(subject, predicate, object, object.getURI(),true,false);
        String queryFilter = this.getQueryPredUriByVar(new Node_Variable("sub"), predicate, new Node_Variable("obj"), "obj");
//        HashMap<String,String> qFilterDict = new HashMap<>();
//        qFilterDict.put("type","uri");
//        qFilterDict.put("on","obj");
//        qFilterDict.put("query", queryFilter);
//        predQueryToHist.put(predicate.getURI(),qFilterDict);
        addQueryFilter("uri","obj", queryFilter, predicate.getURI());

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
        HashMap<String, Object> pred =  this.createPredMap(subject, predicate, object, subject.getURI(),false, true);
        String queryFilter = this.getQueryPredUriByVar(new Node_Variable("sub"), predicate, new Node_Variable("obj"), "sub");
//        HashMap<String,String> qFilterDict = new HashMap<>();
//        qFilterDict.put("type","uri");
//        qFilterDict.put("on","sub");
//        qFilterDict.put("query", queryFilter);
//        predQueryToHist.put(predicate.getURI(),qFilterDict);
        addQueryFilter("uri","sub", queryFilter, predicate.getURI());
        this.queryPredicatesUris.add(pred);
    }

    public ArrayList<ArrayList> getRepresentationJoinsLite(HashMap<String, HashMap<String, ArrayList<String>>> queryJoinsNodes) {
        ArrayList<ArrayList> joisRep = new ArrayList<>();
        ArrayList<String> keys =  new ArrayList<>(queryJoinsNodes.keySet());
        for (int i = 0; i <keys.size() ; i++) {
            HashMap<String, ArrayList<String>> stringArrayListHashMap = queryJoinsNodes.get(keys.get(i));
            ArrayList<String> incoming = stringArrayListHashMap.get("incoming");
            incoming.addAll(stringArrayListHashMap.get("incoming_uri"));
            ArrayList<String> outgoing = stringArrayListHashMap.get("outgoing");
            outgoing.addAll(stringArrayListHashMap.get("outgoing_uri"));
            // Processing incomings...
            for (int j = 0; j < incoming.size(); j++) {
                // Incoming -> node -> outgoings
                for (int k = 0; k < outgoing.size(); k++) {
                    ArrayList<String> trainSample = new ArrayList<>();
                    trainSample.add(incoming.get(j));
                    trainSample.add(outgoing.get(k));
                    trainSample.add(String.valueOf(QueryFeatureExtractor.ONE_WAY_TWO_PREDS));
                    joisRep.add(trainSample);
                }
                // Incoming --> node <-- Incoming
                for (int k = 0; k < incoming.size(); k++) {
                    String predA = incoming.get(j);
                    String predB = incoming.get(k);
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
            for (int j = 0; j < outgoing.size(); j++) {
                // VAR <--OUTGOING <-- node --> OUTGOING --> VAR
                String predA = outgoing.get(j);
                for (int k = 0; k < outgoing.size(); k++) {
                    String predB = outgoing.get(k);
                    //If are diferents then add to list
                    if(!predA.equals(predB)){
                        ArrayList<String> trainSample = new ArrayList<>();
                        trainSample.add(predA);
                        trainSample.add(predB);
                        trainSample.add(String.valueOf(QueryFeatureExtractor.TWO_OUTGOING_PREDS));
                        joisRep.add(trainSample);
                    }
                }
            }
        }
        return joisRep;
    }

    public ArrayList<ArrayList> getRepresentationJoins(HashMap<String, HashMap<String, ArrayList<String>>> queryJoinsNodes) {
        ArrayList<ArrayList> joisRep = new ArrayList<>();
        ArrayList<String> keys =  new ArrayList<>(queryJoinsNodes.keySet());
        for (int i = 0; i <keys.size() ; i++) {
            HashMap<String, ArrayList<String>> stringArrayListHashMap = queryJoinsNodes.get(keys.get(i));
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
        String s = "PREFIX foaf: <http://xmlns.com/foaf/0.1/> SELECT ?name ?email WHERE {  ?x foaf:knows ?y . ?y foaf:name ?name .  OPTIONAL { ?y foaf:mbox ?email } " +
                "FILTER (10 <= ?email) }";
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
            System.out.println("###################################################");
            RLQueryFeatureExtractor qfe = new RLQueryFeatureExtractor(query);
            Map<String, Object> data = qfe.getProcessedData();
            System.out.println(data);
            System.out.println("###################################################");
        }

//        String a = "select (MIN(?val) as ?min) (MAX(?val) as ?max) where {\n" +
//                "  ?numeric <http://dbpedia.org/ontology/apoapsis>  ?val }";
//        Query query = QueryFactory.create(a);
//        // Generate algebra
//        Op op = Algebra.compile(query);
//        op = Algebra.optimize(op);

    }
}
