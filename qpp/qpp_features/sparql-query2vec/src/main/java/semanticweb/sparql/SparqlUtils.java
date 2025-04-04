package semanticweb.sparql;

import liquibase.util.csv.opencsv.CSVReader;
import nanoxml.XMLElement;
import org.apache.commons.io.IOUtils;
import org.apache.jena.base.Sys;
import org.apache.jena.datatypes.xsd.XSDDateTime;
import org.apache.jena.graph.Node;
import org.apache.jena.graph.NodeFactory;
import org.apache.jena.graph.Node_URI;
import org.apache.jena.graph.Triple;
import org.apache.jena.query.*;
import org.apache.jena.rdf.model.*;
import org.apache.jena.sparql.algebra.Algebra;
import org.apache.jena.sparql.core.TriplePath;
import org.apache.jena.sparql.function.library.triple.TriplePredicate;
import org.apache.jena.sparql.path.P_Seq;
import org.apache.jena.sparql.syntax.*;
import semanticweb.EditDistanceAction;
import semanticweb.GraphBuildAction;
import semanticweb.RDF2GXL;
import semanticweb.RDFGraphMatching;
import util.Graph;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.Collectors;

public class SparqlUtils {


    final public static String SPARQL_VAR_NS = "http://wimmics.inria.fr/kolflow/qp#";
    public static Model model;
    public static String prefixes = "";
    public static ArrayList<String[]> queriesError = new ArrayList<>();

    public static void getPropsAndObjectCount() {
//		Map<String, Integer> map = new HashMap<String, Integer>();
        ParameterizedSparqlString qs = new ParameterizedSparqlString("" +
                "select distinct ?property where {\n"
                + "  ?subject ?property ?object . \n" +
                "}");

        QueryExecution exec = QueryExecutionFactory.sparqlService("http://dbpedia.org/sparql", qs.asQuery());

        ResultSet results = exec.execSelect();
        try {
            BufferedWriter br = new BufferedWriter(new FileWriter("properties1.txt"));
            BufferedWriter prop_count = new BufferedWriter(new FileWriter("properties_count1.csv"));

            StringBuilder sb = new StringBuilder();
            StringBuilder sb2 = new StringBuilder();
            while (results.hasNext()) {
                String prop = results.next().get("property").toString();
                sb.append("<" + prop + ">");
                sb.append("\n");
                sb2.append("<" + prop + ">");
                sb2.append(",");
                sb2.append(getObjectCount(prop));
                sb2.append("\n");
            }
            br.write(sb.toString());
            br.close();
            prop_count.write(sb2.toString());
            prop_count.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
//		System.out.println(map.toString());
    }

    /**
     * Retrieve array list of csv file, using delimiter  column param
     *
     * @param url String File path to extract data.
     * @param delimiterCol String delimiter of columns.
     * @return ArrayList with data
     */
    public static ArrayList<String[]> getArrayFromCsvFile(String url, String delimiterCol) {
        BufferedReader csvReader;
        String row;
        ArrayList<String[]> arrayList = new ArrayList<>();
        try {
            csvReader = new BufferedReader(new FileReader(url));
            while ((row = csvReader.readLine()) != null) {
                String[] data = row.split(delimiterCol);
                arrayList.add(data);
            }
            csvReader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return arrayList;
    }

    /**
     * Retrieve array list of csv file, using delimiter  column param
     *
     * @param url String File path to extract data.
     * @param delimiterCol String delimiter of columns.
     * @param delimiterRow String delimiter of rows.
     * @return ArrayList with data
     */
    public static ArrayList<String[]> getArrayFromCsvFile(String url, String delimiterCol, String delimiterRow) {
        BufferedReader csvReader;
        String row;
        ArrayList<String[]> arrayList = new ArrayList<>();
        try {
            csvReader = new BufferedReader(new FileReader(url));
            while ((row = csvReader.readLine()) != null) {
                String[] data = row.split(delimiterCol);
                arrayList.add(data);
            }
            csvReader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return arrayList;
    }

    public static Integer countQueryInProcess = 0;

    /**
     * Retrieve list of predicates array in file
     *
     * @param url String
     * @return Array of array list of strings
     */
    public static ArrayList<ArrayList<String>> getArrayQueries(String url) {
        String row;
        ArrayList<ArrayList<String>> arrayList = new ArrayList<ArrayList<String>>();
        try {
            BufferedReader csvReader = new BufferedReader(new FileReader(url));
            while ((row = csvReader.readLine()) != null) {
                countQueryInProcess++;
                ArrayList<String> predicates = retrievePredicatesInTriples(row);
                arrayList.add(predicates);
            }
            csvReader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return arrayList;
    }

    /**
     * Retrieve list of predicates array in file
     *
     * @param url String
     * @return Array of array list of strings
     */
    public static ArrayList<ArrayList<String>> getArrayQueriesFromCsv(String url, boolean header, int queryColumn, int idColumn) {
        String row;
        ArrayList<ArrayList<String>> arrayList = new ArrayList<ArrayList<String>>();
        int count = 0;
        try {
            BufferedReader csvReader = new BufferedReader(new FileReader(url));
            if (header) {
                //Ignore first read that corresponde with header
                csvReader.readLine();
            }
            while ((row = csvReader.readLine()) != null) {
                countQueryInProcess++;
                String[] rowArray = row.split(",");
                row = rowArray[queryColumn];
                //Remove quotes in init and end of the string...
                row = row.replaceAll("^\"|\"$", "");
                ArrayList<String> predicatesAndId = new ArrayList<>();
                if (idColumn >= 0)
                    predicatesAndId.add(rowArray[idColumn]);
                predicatesAndId.addAll(retrievePredicatesInTriples(row));
                arrayList.add(predicatesAndId);
            }
            csvReader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return arrayList;
    }

    public static ArrayList<String[]> getArrayFromCsvFile(String url) {

        String delimiterCol = ",";
        String delimiterRow = "\n";
        return getArrayFromCsvFile(url, delimiterCol, delimiterRow);
    }

    /**
     *
     * @param url File url to extract namespaces
     * @return HashMap with
     */
    public static HashMap<String, String> getNamespacesStr(String url) {
        String prefixes = "";
        Model model = ModelFactory.createDefaultModel();
        try {
            BufferedReader csvReader = new BufferedReader(new FileReader(url));
            String row;
            while ((row = csvReader.readLine()) != null) {
                String[] predicates = row.split("\t");
                model.setNsPrefix(predicates[0], predicates[1]);
                prefixes = prefixes.concat("PREFIX ").concat(predicates[0]).concat(": ").concat("<").concat(predicates[1]).concat("> \n");
            }
            csvReader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return (HashMap<String, String>) model.getNsPrefixMap();
    }

    /**
     * Get namespaces from file.
     * @param url File url to extract namespaces
     * @return Model object of Jena that contains namespaces.
     */
    public static Model getNamespaces(String url) {

        Model model = ModelFactory.createDefaultModel();
        try {
            BufferedReader csvReader = new BufferedReader(new FileReader(url));
            String row;
            while ((row = csvReader.readLine()) != null) {
                String[] predicates = row.split("\t");
                model.setNsPrefix(predicates[0], predicates[1]);
                prefixes = prefixes.concat("PREFIX ").concat(predicates[0]).concat(": ").concat("<").concat(predicates[1]).concat("> \n");
            }
            csvReader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return model;
    }
    public static Model getNamespacesFromCsv( String url, String delimiter) {
        Model model = ModelFactory.createDefaultModel();
        return getNamespacesFromCsv(model, url, delimiter);
    }
    /**
     * Get namespaces from file.
     * @param url File url to extract namespaces
     * @return Model object of Jena that contains namespaces.
     */
    public static Model getNamespacesFromCsv(Model model, String url, String delimiter) {
        try {
            BufferedReader csvReader = new BufferedReader(new FileReader(url));
            String row;
            while ((row = csvReader.readLine()) != null) {
                String[] predicates = row.split(delimiter);
                model.setNsPrefix(predicates[0], predicates[1]);
                prefixes = prefixes.concat("PREFIX ").concat(predicates[0]).concat(": ").concat("<").concat(predicates[1]).concat("> \n");
            }
            csvReader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return model;
    }

    /**
     * Read a file with prefix in format:
     * PREFIX b3s: <http://b3s.openlinksw.com/>
     * PREFIX b3s2: <http://b3s.openlinksw2.com/>
     *
     * @return a {@link Model} object with the prefix loaded
     */
    public static Model getNamespacesDBPed(Model model, String url) {
        try {
            BufferedReader csvReader = new BufferedReader(new FileReader(url));
            String row;
            while ((row = csvReader.readLine()) != null) {
                String[] predicates = row.split(" ", 2)[1].split(":", 2);
                model.setNsPrefix(predicates[0], predicates[1].replaceAll(" ", "").replace("<", "").replace(">", ""));
                prefixes = prefixes.concat(row).concat("\n");
            }
            csvReader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return model;
    }

    /**
     * Read a file with prefix in format:
     * PREFIX b3s: <http://b3s.openlinksw.com/>
     * PREFIX b3s2: <http://b3s.openlinksw2.com/>
     *
     * @return a {@link Model} object with the prefix loaded
     */
    public static Model getNamespacesDBPed(String url) {
        Model model = ModelFactory.createDefaultModel();
        return getNamespacesDBPed(model, url);
    }

    public static HashMap<String, Integer> readQueryTPFSampling(String route, Character separator) throws IOException {
        InputStreamReader csv = new InputStreamReader(new FileInputStream(route));
        CSVReader csvReader = new CSVReader(csv, separator);
        String[] record;
        HashMap<String, Integer> tpfSamplings = new HashMap<>();
        int count = 0;
        while ((record = csvReader.readNext()) != null) {
            if (record[0].equals("")) {
                continue;
            }
            count++;
            System.out.println(count);
            tpfSamplings.put(record[0].split("/")[8], Integer.parseInt(record[1].replaceAll(" ", "")));
        }
        return tpfSamplings;
    }

    /**
     * @return
     */
    public static Model getNamespaces() {
        String url = "Sparql2vec/prefixes.txt";
        return getNamespaces(url);
    }

    /**
     * Retrieve array of queries in vectors way
     *
     * @param urlQueries File url to read queries.
     * @param urlFeatures File url to read features by line.
     * return ArrayList queries.
     */
    public static ArrayList<String[]> getArrayFeaturesVector(String urlQueries, String urlFeatures, String namespaces, String output) {

        model = getNamespacesDBPed(namespaces);

        ArrayList<String[]> vectors = new ArrayList<String[]>();

        //Get features list, in [0] uri, in [1] frequency
        ArrayList<String[]> featuresArray = getArrayFromCsvFile(urlFeatures);
        Map<String, Integer> featuresMap = new HashMap<>();
        ArrayList<ArrayList<String>> featInQueryList = getArrayQueriesFromCsv(urlQueries, true, 1, 0);
        //we use the size of array intead of -1(csv header) because we use extra column called others.
        String[] vectorheader = new String[featuresArray.size() + 2];
        vectorheader[0] = "id";
        vectorheader[1] = "OTHER";
        int i = 2;
        while (i < featuresArray.size()) {
            featuresMap.put(featuresArray.get(i)[0], i);
            vectorheader[i] = featuresArray.get(i)[0];
            i++;
        }

//        produceCsvArray2(featInQueryList,output);
        for (ArrayList<String> queryArr : featInQueryList) {
            String[] vector = new String[vectorheader.length];
            boolean idSeted = false;
            for (String s : queryArr) {
                try {
                    if (!idSeted) {
                        idSeted = true;
                        vector[0] = s;
                        continue;
                    }
                    int index = featuresMap.get("<" + s + ">");
                    if (vector[index] == null)
                        vector[index] = String.valueOf('0');
                    vector[index] = String.valueOf(Integer.parseInt(vector[index]) + 1);
                } catch (Exception ex) {
                    //ex.printStackTrace();
                    if (vector[1] == null)
                        vector[1] = String.valueOf('0');
                    vector[1] = String.valueOf(Integer.parseInt(vector[1]) + 1);
                }

            }
            vectors.add(vector);
        }

        produceCsvArrayVectors(vectorheader, vectors, output);
        return vectors;
    }

    public static String getObjectCount(String property) {
        ParameterizedSparqlString qs = new ParameterizedSparqlString("" +
                "Select (count(?object) as ?count) Where {\n"
                + "  ?subject <" + property + "> ?object . \n" +
                "}");

        QueryExecution exec = QueryExecutionFactory.sparqlService("http://dbpedia.org/sparql", qs.asQuery());

        ResultSet results = exec.execSelect();

        if (results.hasNext())
            return results.next().getLiteral("count").getString();
        else
            return "0";
    }

    public static String clearQuery(String s) {
        return s.replaceAll("&format=(json|xml|html)", "").
                replaceAll("&output=(json|xml|html)", "");
    }

    public static String replacePrefixes(String s) {
        try {
            Query query = QueryFactory.create(prefixes.concat(s));

            query.setPrefixMapping(query.getPrefixMapping());
            return query.serialize();
        } catch (QueryParseException ex) {
            return "";
        }
    }

    public static String fixVariables(String s) {
        return s.replaceAll(" [?|$][a-zA-Z0-9_]+", " ?variable ");
    }

    public static String[] getQueryAsTokens(String s) {

        String query = clearQuery(s);
        query = replacePrefixes(query);
        query = fixVariables(query);
        query = query.
                replaceAll("[\\{\\}\\(\\)( )]+", " ").
                replaceAll("[\n]*", "").
                replaceAll(" \\. ", " ").
                replaceAll(" \\.", " ").
                replaceAll("\\. ", " ").
                toLowerCase();
        return query.split(" ");
    }

    /**
     * Retrive query ready for excecution, is cleaned some dirty elements.
     *
     * @param queryString String
     * @return String
     */
    public static String getQueryReadyForExecution(String queryString) {
        try {
            queryString = java.net.URLDecoder.decode(queryString, StandardCharsets.UTF_8.name());
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }
        queryString = queryString.substring(queryString.toLowerCase().indexOf("query=") + 6);
        queryString = clearQuery(queryString);
        queryString = replacePrefixes(queryString);
        queryString = queryString.replaceAll("[\n]*", "");
        return queryString;
    }

    /**
     * Retrive query ready for excecution, is cleaned some dirty elements.
     *
     * @param queryString String
     * @return String
     */
    public static String getQueryReadyForExecution(String queryString, boolean isCleaned) {

        //If not cleaned the query process based on logs format....
        if (!isCleaned) {
            try {
                queryString = java.net.URLDecoder.decode(queryString, StandardCharsets.UTF_8.name());
            } catch (UnsupportedEncodingException e) {
                e.printStackTrace();
            }
            queryString = queryString.substring(queryString.toLowerCase().indexOf("query=") + 6);
            queryString = clearQuery(queryString);
        }
        queryString = replacePrefixes(queryString);
        queryString = queryString.replaceAll("[\n]*", "");
        return queryString;
    }

    /**
     * Create a csv with array data passed as parameters.
     *
     * @param list
     * @param filepath
     */
    public static void produceCsvArray(ArrayList<String[]> list, String filepath) {
        BufferedWriter br = null;
        try {
            br = new BufferedWriter(new FileWriter(filepath));

            StringBuilder sb = new StringBuilder();
            // Append strings from array
            for (String[] aList : list) {
                for (String element : aList) {
                    sb.append(element);
                    sb.append(";,;");
                }
                sb.append("\n");
            }

            br.write(sb.toString());
            br.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Create a csv with array data passed as parameters.
     *
     * @param list
     * @param filepath
     */
    public static void produceCsvArray2(ArrayList<ArrayList<String>> list, String filepath) {
        BufferedWriter br = null;
        try {
            br = new BufferedWriter(new FileWriter(filepath));

            StringBuilder sb = new StringBuilder();
            // Append strings from array
            for (ArrayList<String> aList : list) {
                for (String element : aList) {
                    sb.append(element);
                    sb.append(";,;");
                }
                sb.append("\n");
            }

            br.write(sb.toString());
            br.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Create a csv with array data passed as parameters.
     *
     * @param list
     * @param filepath
     */
    public static void produceCsvArrayVectors(String[] headers, ArrayList<String[]> list, String filepath) {
        BufferedWriter br = null;
        try {
            br = new BufferedWriter(new FileWriter(filepath));

            StringBuilder sb = new StringBuilder();
            // Append strings from array
            for (String element : headers) {
                sb.append(element);
                sb.append(",");
            }

            sb.append("\n");
            for (String[] aList : list) {
                for (String element : aList) {
                    if (element == null)
                        element = String.valueOf(0);
                    sb.append(element);
                    sb.append(",");
                }
                sb.append("\n");
            }

            br.write(sb.toString());
            br.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Retrieve the set of triples in a sparql query pattern
     *
     * @param s a sparql query
     * @return Set<Triple>
     */
    public static Set<Triple> retrieveTriples(String s) {

        final Set<Triple> allTriples = new HashSet<Triple>();

        // Parse
        final Query query = QueryFactory.create(s);

        Element e = query.getQueryPattern();

        // This will walk through all parts of the query
        ElementWalker.walk(e,
                // For each element...
                new ElementVisitorBase() {
                    // ...when it's a block of triples...
                    public void visit(ElementPathBlock el) {
                        // ...go through all the triples...
                        Iterator<TriplePath> triples = el.patternElts();
                        while (triples.hasNext()) {
                            TriplePath t = triples.next();
                            Triple triple = t.asTriple();
                            if (triple == null){
                                //Es que no es instancia de 1-Link
//                                if (t.getPath() instanceof P_Seq){
                                    triple = Triple.create(t.getSubject(), NodeFactory.createURI(t.getPath().toString()),t.getObject());
//                                }
                            }
                            allTriples.add(triple);
                        }
                    }

                    public void visit(ElementTriplesBlock el) {
                        Iterator<Triple> triples = el.patternElts();
                        while (triples.hasNext()) {
                            Triple t = triples.next();
                            allTriples.add(t);
                        }
                    }

                    public void visit(ElementSubQuery el) {


                        Query sQuery = el.getQuery();
                        sQuery.setPrefixMapping(query.getPrefixMapping());
                        String sQueryStr = el.getQuery().serialize();

                        Set<Triple> triples = retrieveTriples(sQueryStr);
                        allTriples.addAll(triples);

                    }
                }
        );
        return allTriples;

    }

    /**
     * Retrieve predicates list from query string.
     *
     * @param s a sparql query
     * @return Set<Triple>
     */
    public static ArrayList<String> retrievePredicatesInTriples(final String s) {

        final ArrayList<String> predicates = new ArrayList<String>();
        try {
            final Query query = QueryFactory.create(getQueryReadyForExecution(s, true));
            Element e = query.getQueryPattern();

            // This will walk through all parts of the query
            ElementWalker.walk(e,
                    // For each element...
                    new ElementVisitorBase() {
                        // ...when it's a block of triples...
                        public void visit(ElementPathBlock el) {
                            // ...go through all the triples...
                            Iterator<TriplePath> triples = el.patternElts();
                            while (triples.hasNext()) {
                                // ...and grab the subject
                                //subjects.add(triples.next().getSubject());
                                TriplePath t = triples.next();
                                //System.out.println(t.toString());
                                try {
                                    if (!t.getPredicate().isURI())
                                        continue;
                                    predicates.add(t.getPredicate().getURI());
                                } catch (Exception ex) {
                                    String[] a = new String[2];
                                    a[0] = s;
                                    a[1] = String.valueOf(countQueryInProcess);
                                    queriesError.add(a);
                                    ex.printStackTrace();
                                }
                            }
                        }

                        public void visit(ElementTriplesBlock el) {
                            // ...go through all the triples...
                            Iterator<Triple> triples = el.patternElts();
                            while (triples.hasNext()) {
                                // ...and grab the subject
                                //subjects.add(triples.next().getSubject());
                                Triple t = triples.next();
                                //System.out.println(t.toString());
                                predicates.add(t.getPredicate().getURI());
                            }


                        }

                        public void visit(ElementSubQuery el) {

                            Query sQuery = el.getQuery();
                            sQuery.setPrefixMapping(query.getPrefixMapping());
                            String sQueryStr = el.getQuery().serialize();

                            ArrayList<String> predicats = retrievePredicatesInTriples(sQueryStr);
                            predicates.addAll(predicats);
                        }
                    }
            );
            return predicates;
        } catch (Exception ex) {
            return predicates;
        }


    }


    /**
     * Replaces the ? with a URI to hel create an RDF graph with the sparql variables
     *
     * @param symbol name of the variable
     * @return refined String URI for the sparql variable
     */
    private static String refineSymbol(String symbol) {
        if (symbol.contains("?")) {
            symbol = symbol.replaceAll("\\?", SPARQL_VAR_NS);
        }
        return symbol;
    }

    /**
     * Replaces the ? with a URI to hel create an RDF graph with the sparql variables
     *
     * @param node node for the sparql variable
     * @return refined String URI for the sparql variable
     */
    private static String refineSymbol(Node node) {
        return refineSymbol(node.toString());

    }

    /**
     * Builds an RDF graph from the sparql query pattern
     *
     * @param s sparql query string
     * @return an RDF graph from the sparql query pattern
     */

    public static Model buildQueryRDFGraph(String s) {
        // create an empty model
        Model model = ModelFactory.createDefaultModel();

        Set<Triple> triples = retrieveTriples(s);

        for (Triple t : triples) {
            Node sub = t.getSubject();


            Resource rSub = null;

            if (sub.isVariable()) {

                String refineSubURI = refineSymbol(sub);

                rSub = model.createResource(refineSubURI);

            } else {
                rSub = model.asRDFNode(sub).asResource();
            }

            Node pred = t.getPredicate();

            Property rPred = null;

            if (pred.isVariable()) {


                String refinePredUri = refineSymbol(pred);

                rPred = model.createProperty(refinePredUri);
            } else {
                rPred = model.createProperty(pred.toString());

            }


            Node obj = t.getObject();
            RDFNode rObj = null;

            if (obj.isVariable()) {


                String refineObjUri = refineSymbol(obj);

                rObj = model.createResource(refineObjUri);
            } else {
                rObj = model.asRDFNode(obj);
            }

            //System.out.println(rSub.getClass());
            //System.out.println(rPred.getClass());
            //System.out.println(rObj.getClass());

            Statement st = model.createStatement(rSub, rPred, rObj);
            //System.out.println(st);
            model.add(st);

        }
        return model;

    }
    //our implemenetation for learned sparql works
    public static Model buildQueryRDFGraph_other(String s) {
        // create an empty model
        Model model = ModelFactory.createDefaultModel();

        Set<Triple> triples = retrieveTriples(s);

        for (Triple t : triples) {
            Node sub = t.getSubject();

            Resource rSub = null;

            if (sub.isVariable()) {

                String refineSubURI = refineSymbol(sub);

                rSub = model.createResource(refineSubURI);

            } else {
                rSub = model.asRDFNode(sub).asResource();
            }

            Node pred = t.getPredicate();

            Property rPred = null;

            if (pred.isVariable()) {


                String refinePredUri = refineSymbol(pred);

                rPred = model.createProperty(refinePredUri);
            } else {
                rPred = model.createProperty(pred.toString());

            }


            Node obj = t.getObject();
            RDFNode rObj = null;

            if (obj.isVariable()) {


                String refineObjUri = refineSymbol(obj);

                rObj = model.createResource(refineObjUri);
            } else {
                rObj = model.asRDFNode(obj);
            }

            //System.out.println(rSub.getClass());
            //System.out.println(rPred.getClass());
            //System.out.println(rObj.getClass());

            Statement st = model.createStatement(rSub, rPred, rObj);
            //System.out.println(st);
            model.add(st);

        }
        return model;

    }

    /**
     * Builds an RDF graph from the sparql query pattern
     *
     * @param s sparql query string
     * @return an RDF graph from the sparql query pattern
     */

    public static Model buildQueryRDFGraph(String s, Model model) {
        // create an empty model
        Set<Triple> triples = retrieveTriples(s);

        for (Triple t : triples) {
            Node sub = t.getSubject();


            Resource rSub = null;

            if (sub.isVariable()) {

                String refineSubURI = refineSymbol(sub);

                rSub = model.createResource(refineSubURI);

            } else {
                rSub = model.asRDFNode(sub).asResource();
            }

            Node pred = t.getPredicate();

            Property rPred = null;

            if (pred.isVariable()) {


                String refinePredUri = refineSymbol(pred);

                rPred = model.createProperty(refinePredUri);
            } else {
                rPred = model.createProperty(pred.toString());

            }


            Node obj = t.getObject();
            RDFNode rObj = null;

            if (obj.isVariable()) {


                String refineObjUri = refineSymbol(obj);

                rObj = model.createResource(refineObjUri);
            } else {
                rObj = model.asRDFNode(obj);
            }

            //System.out.println(rSub.getClass());
            //System.out.println(rPred.getClass());
            //System.out.println(rObj.getClass());

            Statement st = model.createStatement(rSub, rPred, rObj);
            //System.out.println(st);
            model.add(st);

        }
        return model;

    }

    /**
     * Returns true if the Resource represented by the URI was a variable in the original sparql query
     *
     * @param uri a RDF resource URI
     * @return true or false
     */
    private static boolean wasVariable(String uri) {
        if (uri.contains(SPARQL_VAR_NS)) return true;
        return false;
    }

    /**
     * Builds a GXL graph suitable for the GMT library from a sparql query
     *
     * @param qr      a sparql query
     * @param graphId an id for the query, sometimes useful for indexing
     * @return a representation of the GXL graph
     * @throws Exception
     */

    public static Graph buildSPARQL2GXLGraph(String qr, String graphId) throws Exception {

        Model model = buildQueryRDFGraph(qr);

        XMLElement gxl = RDF2GXL.getGXLRootElement();

        XMLElement graph = RDF2GXL.getGXLGraphElement(graphId);

        gxl.addChild(graph);


        // write it to standard out
        //model.write(System.out);


        ResIterator subIterator = model.listSubjects();
        while (subIterator.hasNext()) {
            Resource sub = subIterator.nextResource();
            XMLElement gxlSub = null;
            if (wasVariable(sub.toString())) {

                gxlSub = RDF2GXL.transformResourceURI2GXL(sub.toString(), "?");

            } else {
                gxlSub = RDF2GXL.transformResourceURI2GXL(sub.toString());
            }

            graph.addChild(gxlSub);
        }

        NodeIterator objIterator = model.listObjects();
        while (objIterator.hasNext()) {
            RDFNode obj = objIterator.nextNode();
            XMLElement gxlObj = null;
            if (wasVariable(obj.toString())) {
                gxlObj = RDF2GXL.transformResourceURI2GXL(obj.toString(), "?");
            } else {
                //check in RDF spec whether literals with same values are considered as same RDF graph nodes.
                gxlObj = RDF2GXL.transformResourceURI2GXL(obj.toString());
            }
            graph.addChild(gxlObj);
        }


        StmtIterator stmtIterator = model.listStatements();

        while (stmtIterator.hasNext()) {
            Statement s = stmtIterator.nextStatement();
            //System.out.println(s);
            String fromURI = s.getSubject().toString();
            String predicateURI = wasVariable(s.getPredicate().toString()) ? "?" : s.getPredicate().toString();
            String toResource = s.getObject().toString();

            XMLElement edge = RDF2GXL.transformTriple2GXL(fromURI, predicateURI, toResource);
            graph.addChild(edge);


        }

        return RDF2GXL.parseGXL(gxl);


    }

    /**
     * Builds a GXL graph suitable for the GMT library from a sparql query
     *
     * @param qr      a sparql query
     * @param graphId an id for the query, sometimes useful for indexing
     * @return a representation of the GXL graph
     * @throws Exception
     */

    public static Graph buildSPARQL2GXLGraph(String qr, String graphId, Model model) throws Exception {

        model = buildQueryRDFGraph(qr, model);

        XMLElement gxl = RDF2GXL.getGXLRootElement();

        XMLElement graph = RDF2GXL.getGXLGraphElement(graphId);

        gxl.addChild(graph);


        // write it to standard out
        //model.write(System.out);


        ResIterator subIterator = model.listSubjects();
        while (subIterator.hasNext()) {
            Resource sub = subIterator.nextResource();
            XMLElement gxlSub = null;
            if (wasVariable(sub.toString())) {

                gxlSub = RDF2GXL.transformResourceURI2GXL(sub.toString(), "?");

            } else {
                gxlSub = RDF2GXL.transformResourceURI2GXL(sub.toString());
            }

            graph.addChild(gxlSub);
        }

        NodeIterator objIterator = model.listObjects();
        while (objIterator.hasNext()) {
            RDFNode obj = objIterator.nextNode();
            XMLElement gxlObj = null;
            if (wasVariable(obj.toString())) {
                gxlObj = RDF2GXL.transformResourceURI2GXL(obj.toString(), "?");
            } else {
                //check in RDF spec whether literals with same values are considered as same RDF graph nodes.
                gxlObj = RDF2GXL.transformResourceURI2GXL(obj.toString());
            }
            graph.addChild(gxlObj);
        }


        StmtIterator stmtIterator = model.listStatements();

        while (stmtIterator.hasNext()) {
            Statement s = stmtIterator.nextStatement();
            //System.out.println(s);
            String fromURI = s.getSubject().toString();
            String predicateURI = wasVariable(s.getPredicate().toString()) ? "?" : s.getPredicate().toString();
            String toResource = s.getObject().toString();

            XMLElement edge = RDF2GXL.transformTriple2GXL(fromURI, predicateURI, toResource);
            graph.addChild(edge);


        }

        return RDF2GXL.parseGXL(gxl);


    }

    public static ArrayList<String> getQueries(String trainingQueryFile, ArrayList<Integer> not_include) {
        /*Model model = getNamespaces();
        Map<String, String> pref = model.getNsPrefixMap();
        Object[] keys = pref.keySet().toArray();
        boolean header = true;
        ArrayList<String> queries = new ArrayList<String>();
        int index = 0;
        try {
            InputStreamReader csv = new InputStreamReader(new FileInputStream(trainingQueryFile));
            CSVReader csvReader = new CSVReader(csv);
            String[] record;
            while ((record = csvReader.readNext()) != null) {
                if (header) {
                    header = false;
                    continue;
                }
                if (not_include.contains(index)) {
                    index++;
                    continue;
                } else {
                    index++;
                }
                String query = record[1].replaceAll("^\"|\"$", "");
                String prefixesStr = "";
                for (int i = 0; i < model.getNsPrefixMap().size(); i++) {

                    int a = query.indexOf(String.valueOf(keys[i] + ":"));
                    if (a != -1) {
                        prefixesStr = prefixesStr.concat("PREFIX ").concat(String.valueOf(keys[i])).concat(": ").concat("<").concat(pref.get(String.valueOf(keys[i]))).concat("> \n");
                    }
                }
                query = prefixesStr.concat(" " + query);
                queries.add(query);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }*/
        Scanner sc = null;
        File f = new File(trainingQueryFile);
        ArrayList<String> queries = new ArrayList<>();
        int unprocessed = 0;
        try {
            sc = new Scanner(f);
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);

        }
        FileWriter writer = null;
        Scanner in = new Scanner(System.in);
        try {

            int iteration = 0;
            while (sc.hasNextLine())
            {
                Scanner linescanner = new Scanner(sc.nextLine());
                linescanner.useDelimiter(",");
                List<String> lineData = new ArrayList<String>();
                linescanner.forEachRemaining(lineData::add);
                if (iteration > 0) {
                    Query q = new Query();
                    String id = lineData.get(0);
                    String queryStr = "";
                    String execTime;
                    if (lineData.size() == 9){
                        queryStr =lineData.get(1);;
                        execTime = lineData.get(7);
                    }else{

                        int queryFrags = lineData.size()-9;
                        for( int i = 1; i<=queryFrags+1;i++){
                            if (i+1 <= queryFrags+1)
                                queryStr = queryStr+ lineData.get(i)+",";
                            else
                                queryStr += lineData.get(i);
                        }
                        //System.out.println("linedata: "+lineData);
                        //System.out.println("querystring: "+queryStr);
                    }


                    try {
                        queryStr = queryStr.replaceAll("^\"|\"$", "");
                        //queryStr= queryStr.replace("\"","");
                        Query queryObj = QueryFactory.create(queryStr, Syntax.syntaxARQ);

                        // Generate algebra
                        Algebra.compile(queryObj);
                        queries.add(queryStr);
                    }
                    catch (Exception exception){

                        //exception.printStackTrace();
                        unprocessed +=1;
                        System.out.println("unprocessed");
                        System.out.println("linedata "+lineData);
                        System.out.println("querystr "+queryStr);
                        //System.out.println("Error with query: ".concat(queryStr).concat(" :<end>"));
                        continue;
                    }
                    //q.type = lineData.get(2);
                    //q.noNotedTriplePatterns = Integer.parseInt( lineData.get(6));
                    //q.runTimeMS = Double.parseDouble(lineData.get(7));
                    //q.resultSize = Integer.parseInt( lineData.get(8));
                    linescanner.close();
                }
                iteration++;
            }
            System.out.println("Amount of unprocessed queries "+unprocessed+"\n");
            sc.close();
            writer.close();
        }catch (IOException e){
            e.printStackTrace();
        }
        return queries;
    }
	public static ArrayList<String[]> getQueries(String trainingQueryFile, String namespaces_path, ArrayList<Integer> not_include, int idColumn, int queryColumn,int execTimeColumn,char input_delimiter, String unprocessed_query){
		System.out.println("Reading Queries..");
        Scanner sc = null;
        File f = new File(trainingQueryFile);
        ArrayList<String[]> queries = new ArrayList<>();
        int unprocessed = 0;
        try {
            sc = new Scanner(f);
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);

        }
        FileWriter writer = null;
        Scanner in = new Scanner(System.in);
        try {
            writer = new FileWriter(unprocessed_query);

        int iteration = 0;
        while (sc.hasNextLine())
        {
            Scanner linescanner = new Scanner(sc.nextLine());
            linescanner.useDelimiter(",");
            List<String> lineData = new ArrayList<String>();
            linescanner.forEachRemaining(lineData::add);
            if (iteration > 0) {
                Query q = new Query();
                //.id = lineData.get(0);
                String id = lineData.get(0);
                String queryStr = "";
                String execTime;
                if (lineData.size() == 9){
                    //System.out.println("In query with queal than 9 in size");
                    //System.out.println(lineData);
                    queryStr =lineData.get(1);;
                    execTime = lineData.get(7);
                }else{
                    //System.out.println("In query with more than 9 in size");

                    int queryFrags = lineData.size()-9;
                    /*int i = 1;
                    while(i<= queryFrags+1){
                        if (i+1 <= queryFrags+1)
                            queryStr = queryStr+ lineData.get(i)+",";
                        else
                            queryStr += lineData.get(i);
                        i++;
                    }*/
                    for( int i = 1; i<=queryFrags+1;i++){
                        if (i+1 <= queryFrags+1)
                            queryStr = queryStr+ lineData.get(i)+",";
                        else
                            queryStr += lineData.get(i);
                    }
                    //System.out.println("linedata: "+lineData);
                    //System.out.println("querystring: "+queryStr);
                    execTime = lineData.get(7+queryFrags);
                }


                try {
                    queryStr = queryStr.replaceAll("^\"|\"$", "");
                    //queryStr= queryStr.replace("\"","");
                    Query queryObj = QueryFactory.create(queryStr, Syntax.syntaxARQ);

                    // Generate algebra
                    Algebra.compile(queryObj);


                    String[] curr = new String[]{id, queryStr, execTime};
                    queries.add(curr);
                }
                catch (Exception exception){

                    exception.printStackTrace();
                    unprocessed +=1;
                    writer.write(queryStr+","+id+","+execTime+ "\n");
                    System.out.println("unprocessed");
                    System.out.println("linedata "+lineData);
                    System.out.println("querystr "+queryStr);
                    //System.out.println("Error with query: ".concat(queryStr).concat(" :<end>"));
                    continue;
                }
                //q.type = lineData.get(2);
                //q.noNotedTriplePatterns = Integer.parseInt( lineData.get(6));
                //q.runTimeMS = Double.parseDouble(lineData.get(7));
                //q.resultSize = Integer.parseInt( lineData.get(8));
                linescanner.close();
            }
            iteration++;
        }
        System.out.println("Amount of unprocessed queries "+unprocessed+"\n");
        sc.close();
            writer.close();
        }catch (IOException e){
            e.printStackTrace();
        }
        return queries;
	}

    public void calculateEditDistance(String input, String output, String prefixFile, int cores, char input_delimiter, char output_delimiter, int idColumn, int queryColumn, int execTimeColumn, int elemtByCore) {
        ArrayList<String[]> queries = SparqlUtils.getQueries(input, prefixFile, new ArrayList<>(), idColumn, queryColumn, execTimeColumn, input_delimiter,"/test/unprocessed_queries.csv");

        System.out.println("Creating Query graphs..");
        ForkJoinPool pool = new ForkJoinPool(cores);
//        GraphBuildAction task = new GraphBuildAction(queries, 0, 50);
        GraphBuildAction task = new GraphBuildAction(queries, 0, queries.size());
        ArrayList<HashMap<String, Object>> grafos = pool.invoke(task);
        System.out.println("Query graphs created" + grafos.size());
        EditDistanceAction task2 = new EditDistanceAction(grafos, output, cores, 0, grafos.size(), elemtByCore);
        pool.invoke(task2);

        String a = "";
    }
    public void calculateEditDistance(ArrayList<lsq_extract.util.Query> queries, String output, int cores, char output_delimiter, int elemtByCore) {
        //ArrayList<String[]> queries = SparqlUtils.getQueries(input, prefixFile, new ArrayList<>(), idColumn, queryColumn, execTimeColumn, input_delimiter,"/test/unprocessed_queries.csv");
        ArrayList<String[]> queryStrs = new ArrayList<>();
        for(lsq_extract.util.Query i : queries){
            queryStrs.add(new String[]{i.id, i.text, String.valueOf( i.duration)});
        }
        System.out.println("Creating Query graphs..");
        ForkJoinPool pool = new ForkJoinPool(cores);
        GraphBuildAction task = new GraphBuildAction(queryStrs, 0, queryStrs.size());
        ArrayList<HashMap<String, Object>> grafos = pool.invoke(task);
        System.out.println("Query graphs created" + grafos.size());
        EditDistanceAction task2 = new EditDistanceAction(grafos, output, cores, 0, grafos.size(), elemtByCore);
        pool.invoke(task2);
    }
    
    public void calculateEditDistance(ArrayList<lsq_extract.util.Query> queries, String output, int cores, char output_delimiter, int elemtByCore, int index) {
        //ArrayList<String[]> queries = SparqlUtils.getQueries(input, prefixFile, new ArrayList<>(), idColumn, queryColumn, execTimeColumn, input_delimiter,"/test/unprocessed_queries.csv");
        ArrayList<String[]> queryStrs = new ArrayList<>();
        for(lsq_extract.util.Query i : queries){
            queryStrs.add(new String[]{i.id, i.text, String.valueOf( i.duration)});
        }
        System.out.println("Creating Query graphs..");
        ForkJoinPool pool = new ForkJoinPool(cores);
        GraphBuildAction task = new GraphBuildAction(queryStrs, 0, queryStrs.size());
        ArrayList<HashMap<String, Object>> grafos = pool.invoke(task);
        System.out.println("Query graphs created" + grafos.size());
        EditDistanceAction task2 = new EditDistanceAction(grafos, output, cores, index, grafos.size(), elemtByCore);
        pool.invoke(task2);
    }
    public void calculateEditDistance(ArrayList<lsq_extract.util.Query> queries, String output, int cores, char output_delimiter, int elemtByCore, int indexStart, int indexEnd) {
        //ArrayList<String[]> queries = SparqlUtils.getQueries(input, prefixFile, new ArrayList<>(), idColumn, queryColumn, execTimeColumn, input_delimiter,"/test/unprocessed_queries.csv");
        ArrayList<String[]> queryStrs = new ArrayList<>();
        for(lsq_extract.util.Query i : queries){
            queryStrs.add(new String[]{i.id, i.text, String.valueOf( i.duration)});
        }
        System.out.println("Creating Query graphs..");
        ForkJoinPool pool = new ForkJoinPool(cores);
        
        GraphBuildAction task = new GraphBuildAction(queryStrs,0,queryStrs.size());
        ArrayList<HashMap<String, Object>> grafos = pool.invoke(task);
        System.out.println("Query graphs created" + grafos.size());
        if (grafos.size() < indexEnd){
            indexEnd = grafos.size();
        }
        EditDistanceAction task2 = new EditDistanceAction(grafos, output, cores, indexStart, indexEnd, elemtByCore);
        pool.invoke(task2);
    }

    public void calculateEditDistanceTrainQueries(ArrayList<String> centerQueries, ArrayList<lsq_extract.util.Query> queries, String outputPath) {
        RDFGraphMatching matcher = new RDFGraphMatching();
        try{
            boolean fileExists = new File(outputPath).exists();
            FileWriter outputWriter = new FileWriter(outputPath);
            if (!fileExists){
                outputWriter.write("id,vec\n");
            }
            ArrayList<String[]> queryStrs = new ArrayList<>();
            for(lsq_extract.util.Query i : queries){
                queryStrs.add(new String[]{i.id, i.text, String.valueOf( i.duration)});
            }
            ArrayList<Graph> centerGraphs = new ArrayList<>();
            for(int query_i = 0; query_i < centerQueries.size(); query_i++){
                try {
                    centerGraphs.add(SparqlUtils.buildSPARQL2GXLGraph(centerQueries.get(query_i),  Integer.toString(query_i)));
                } catch (Exception e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
            }
            
            for(lsq_extract.util.Query q : queries){
                try {
                String q_dists = "";
                q_dists += (q.id +',');
                Graph q_graph;
                        q_graph = SparqlUtils.buildSPARQL2GXLGraph(q.text,  q.id);
                        for (Graph center_g: centerGraphs){
                            q_dists += (Double.toString( matcher.distanceBipartiteHungarian(q_graph, center_g)) +',');
                        }
                        
                        q_dists = q_dists.substring(0,q_dists.length()-1);
                        outputWriter.write(q_dists);
                        outputWriter.write("\n");
                        
                    } catch (Exception e) {
                        // TODO Auto-generated catch block
                        e.printStackTrace();
                    }
            }
            outputWriter.close();
        } catch(IOException e){
            System.out.println("Outputfile could no be written to");
            e.printStackTrace();
        }
        
    }

    private String getGraphPatternByQuery(String[] query, HashMap<String,Graph> medoids, Character output_delimiter){
        Iterator<Map.Entry<String, Graph>> iterator = medoids.entrySet().iterator();

        StringBuilder stringBuilder = new StringBuilder();
        try {
            Graph Gi = SparqlUtils.buildSPARQL2GXLGraph(query[1],  "row_"+ query[0]);
            stringBuilder.append(query[0]).append(output_delimiter);
            stringBuilder.append(query[2]);
            RDFGraphMatching matcher = new RDFGraphMatching();
            while (iterator.hasNext()) {
                Map.Entry<String, Graph> mapElement = iterator.next();
                Graph centroid = ((Graph)mapElement.getValue());
                double dist = matcher.distanceBipartiteHungarian(Gi, centroid);
                stringBuilder.append(output_delimiter);
                double similarity = 1 / (1+ dist);
                stringBuilder.append(similarity);
            }
            stringBuilder.append("\n");
            return stringBuilder.toString();
        }
        catch (Exception ex){
            ex.printStackTrace();
        }
        return null;
    }

    private String getGraphPatternByQuerySingle(String query, HashMap<String,Graph> medoids, Character output_delimiter){
        Iterator<Map.Entry<String, Graph>> iterator = medoids.entrySet().iterator();

        ArrayList<String> features = new ArrayList<>();
        try {
            Graph Gi = SparqlUtils.buildSPARQL2GXLGraph(query,  "row_");
            RDFGraphMatching matcher = new RDFGraphMatching();
            while (iterator.hasNext()) {
                Map.Entry<String, Graph> mapElement = iterator.next();
                Graph centroid = mapElement.getValue();
                double dist = matcher.distanceBipartiteHungarian(Gi, centroid);

                double similarity = 1 / (1+ dist);
                features.add(String.valueOf(similarity));
            }
            return String.join(output_delimiter.toString(), features).concat("\n");

        }
        catch (Exception ex){
            ex.printStackTrace();
        }
        return null;
    }

    public void getGraphPatterns(String input, String output, String modeidsFile, String prefixFile, char input_delimiter, char output_delimiter, int idColumn, int queryColumn, int execTimeColumn) {
        ArrayList<String[]> queries = SparqlUtils.getQueries(input, prefixFile, new ArrayList<>(), idColumn, queryColumn, execTimeColumn, input_delimiter,"./unprocessed.txt");
        System.out.println("Queries readed: ".concat(String.valueOf(queries.size())).concat(" in total."));
        ArrayList<String> medoidsId = new ArrayList<>();
        HashMap<String,Graph> medoidsMap = new HashMap<>();
        BufferedReader reader;
        try {
            reader = new BufferedReader(new FileReader(modeidsFile));
            String line = reader.readLine();
            while (line != null) {
                System.out.println(line);
                medoidsId.add(line.replaceAll("\\s+",""));
                line = reader.readLine();
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        for (String[] query : queries) {
            int index = medoidsId.indexOf(query[idColumn].replaceAll("\\s+",""));
            if (index >= 0) {
                try {
                    Graph Gi = SparqlUtils.buildSPARQL2GXLGraph(query[1],  "row_"+ query[0]);
                    medoidsMap.put(query[idColumn], Gi);
                }
                catch (Exception ex){
                    ex.printStackTrace();
                }

            }
        }
        if(medoidsMap.size() != medoidsId.size()){
            System.out.println("Some Medoids not finded!!! :(");
            return;
        }
        System.out.println("Moedoids readed: ".concat(String.valueOf(medoidsId.size())).concat(" in total."));
        System.out.println("Creating Query graphs..");
        
        List<String> results = queries.parallelStream()
                .map(i -> getGraphPatternByQuery(i, medoidsMap, output_delimiter)) // each operation takes one second
                .collect(Collectors.toList());
        StringBuilder sb = new StringBuilder();
//
        BufferedWriter br;
        try {
            br = new BufferedWriter(new FileWriter(output));
            //Write header
            sb.append("id").append(output_delimiter);
            sb.append("time");
            for (int i = 0; i < medoidsId.size(); i++) {
                sb.append(output_delimiter);
                sb.append("pcs").append(i);
            }
            sb.append("\n");
            for (String result : results) {
                sb.append(result);
                sb.append("\n");
            }
            br.write(sb.toString());
            br.close();
            System.out.println("Medoids vectors computed, output writed in :" + output);
        }
        catch (Exception ex){
            ex.printStackTrace();
            System.out.println("Something was wrong in the writing process of the output");

        }
    }

    public void getQueryGraphPatterns(String input, String output, String modeidsFile, String prefixFile, char input_delimiter, char output_delimiter, int idColumn, int queryColumn, int execTimeColumn) throws Exception {
        ArrayList<String[]> queries = SparqlUtils.getQueries(input, prefixFile, new ArrayList<>(), idColumn, queryColumn, execTimeColumn, input_delimiter,"./unprocessed.txt");
        System.out.println("Queries readed: ".concat(String.valueOf(queries.size())).concat(" in total."));

        HashMap<String,Graph> medoidsMap = new HashMap<>();
        BufferedReader reader;

        try {
            reader = new BufferedReader(new FileReader(modeidsFile));

            String header = reader.readLine();
            String line = reader.readLine();
            int index = 0;
            while (line != null) {

                System.out.println(line);
                String[] medoid_cols = line.split(String.valueOf(input_delimiter));
                String query = medoid_cols[queryColumn].replaceAll("\\s+","");
                Graph Gi = SparqlUtils.buildSPARQL2GXLGraph(query,  "row_"+ index);
                medoidsMap.put(medoid_cols[idColumn], Gi);
                line = reader.readLine();
                index+=1;
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println(queries.get(0)[queryColumn]);
        String vectorgp = getGraphPatternByQuerySingle(queries.get(0)[queryColumn], medoidsMap, output_delimiter);
        String query_id = queries.get(0)[idColumn];
        StringBuilder sb = new StringBuilder();
//
        BufferedWriter br;
        try {
            br = new BufferedWriter(new FileWriter(output));
            //Write header
            sb.append("id").append(output_delimiter);
            sb.append("time");
            for (int i = 0; i < medoidsMap.size(); i++) {
                sb.append(output_delimiter);
                sb.append("pcs").append(i);
            }
            sb.append("\n");
            sb.append(query_id).append(output_delimiter).append("0.0").append(output_delimiter);
            sb.append(vectorgp);
            sb.append("\n");
            br.write(sb.toString());
            br.close();
            System.out.println("Medoids vectors computed, output writed in :" + output);
        }
        catch (Exception ex){
            ex.printStackTrace();
            System.out.println("Something was wrong in the writing process of the output");

        }
    }
    /**
     *aw
     * @param defaultGraph
     * @param output
     * @param init
     * @param last
     * @param limit
     */
    public static void extractDataFromLsq(String defaultGraph, String output, int init, int last, int limit ){
        String qs =
                "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>" +
                        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " +
                        "PREFIX lsqr: <http://lsq.aksw.org/res/>  " +
                        "PREFIX lsqv: <http://lsq.aksw.org/vocab#>  " +
                        "PREFIX sp: <http://spinrdf.org/sp#>  " +
                        "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>  " +
                        "PREFIX purl: <http://purl.org/dc/terms/>  \n" +
                        "\n" +
                        "SELECT  Distinct ?s ?execution  ?issued (xsd:dateTime(?issued) as ?date) ?runTimeMs ?resultSize ?query WHERE {  \n" +
                        " ?s  lsqv:execution ?execution . \n" +
                        "  ?execution purl:issued  ?issued  . \n" +
                        "?s  lsqv:resultSize ?resultSize     .\n" +
                        "\t  ?s  lsqv:runTimeMs ?runTimeMs     .\n" +
                        "\t  ?s  rdf:type ?type     .\n" +
                        "\t  ?s  sp:text ?query     .\n" +
                        "FILTER ( ?runTimeMs >= ".concat(String.valueOf(init)).concat(" && ?runTimeMs < ").concat(String.valueOf(last)).concat("  && ?resultSize >= 0) }\n")
                                .concat("LIMIT ".concat(String.valueOf(limit)));
        ARQ.init();
        Query query = QueryFactory.create(qs) ;
        QueryExecution exec = QueryExecutionFactory.sparqlService("http://localhost:8890/sparql", query, null,null);

        ResultSet results = exec.execSelect();
        try {
            BufferedWriter prop_count = new BufferedWriter(new FileWriter("qeries_lsq_".concat(output).concat("_").concat(String.valueOf(init).concat("_").concat(String.valueOf(last))).concat(".csv")));
            StringBuilder sb2 = new StringBuilder();
            String separator = "ᶶ";
            /*sb2.append("id");
            sb2.append(separator);
            sb2.append("execution");
            sb2.append(separator);
            sb2.append("date");
            sb2.append(separator);
            sb2.append("year");
            sb2.append(separator);
            sb2.append("month");
            sb2.append(separator);
            sb2.append("day");
            sb2.append(separator);
            sb2.append("time");
            sb2.append(separator);
            sb2.append("runtime");
            sb2.append(LineSeparator.Unix);*/
            //sᶶexecutionᶶdateᶶyearᶶmonthᶶdayᶶtimeᶶrunTimeMsᶶresultSizeᶶquery
            while (results.hasNext()) {
                QuerySolution a = results.next();
                //?s ?execution  ?issued ?runTimeMs ?resultSize ?query
                String s = String.valueOf(a.get("s"));
                sb2.append(s);
                sb2.append(separator);

                String execution = String.valueOf(a.get("execution"));
                sb2.append(execution);
                sb2.append(separator);

                String date = String.valueOf(a.get("date"));
                sb2.append(date);
                sb2.append(separator);

                XSDDateTime issued = ((XSDDateTime) a.getLiteral("issued").getValue());
                int year = issued.getYears();
                sb2.append(year);
                sb2.append(separator);

                int month = issued.getMonths();
                sb2.append(month);
                sb2.append(separator);

                int day = issued.getDays();
                sb2.append(day);
                sb2.append(separator);

                double time = issued.getTimePart();
                sb2.append(time);
                sb2.append(separator);

                double runtime = a.getLiteral("runTimeMs").getDouble();
                sb2.append(runtime);
                sb2.append(separator);

                double resultSize = a.getLiteral("resultSize").getDouble();
                sb2.append(resultSize);
                sb2.append(separator);

                String queryLog = String.valueOf(a.get("query"));
                sb2.append(queryLog.replaceAll("\t", " ").replaceAll("\r"," ").replaceAll("\n"," "));
                sb2.append(IOUtils.LINE_SEPARATOR_UNIX);  //Esto porque es el último valor de la fila.
            }
            prop_count.write(sb2.toString());
            prop_count.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}