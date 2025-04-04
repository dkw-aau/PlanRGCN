package  com.org;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.concurrent.TimeUnit;

import com.org.Algebra.EntityLogExtractor;
import com.org.Algebra.PredicateLogExtractor;
import com.org.Algebra.Utils;
import com.org.QueryReader.LSQreader;

import org.apache.commons.lang3.time.StopWatch;
import org.apache.jena.query.Query;
import org.apache.jena.query.QueryFactory;
import org.apache.jena.sparql.algebra.Algebra;
import org.apache.jena.sparql.algebra.Op;
import com.org.ex_plan.QueryGraphVisitor;
import java.io.PrintStream;
import java.io.FileNotFoundException;

public class App
{
    public static void main( String[] args )
    {
        String task = args[0];
        switch (task){
        case "qg": {
            try{
            System.setErr(new PrintStream("/dev/null"));
            }catch (FileNotFoundException e){
            }

            printQueryGraph(args[1]);

            break;
        }
            case "test": {
                test();
                break;
            }
	    case "extract-single":{
	    	Utils u = new Utils();
		u.create_algebra_singleQuery(args[1],args[2]);
		break;
	    }
            case "extract-query-plans":{
                try {
                    LSQreader reader = new LSQreader(args[1]);
                    
                    Utils u = new Utils();
                    if (args[3].equals("lsq=true")){
                        Utils.sub_id = true;
                    }
                    
                    u.extract_query_plan(reader, args[2]);
                } catch (FileNotFoundException e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
                break;
            }
            case "extract-predicates-query-log":{
                PredicateLogExtractor.run(args[1], args[2]);
                break;
            }
            case "extract-entity-query-log":{
                EntityLogExtractor.run(args[1], args[2]);
                break;
            }
            case "time-query-plan-gen": {
                Utils u = new Utils();
                if (args[3].equals("lsq=true")) {
                    Utils.sub_id = true;
                }

                try {
                    u.time_query_plan_extraction(args[1], args[2]);
                } catch (IOException e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
                break;
            }
            default: {
                System.out.println("Something went wrong with the task specifications");
                break;
            }
        }
        
    }

    public static void test(){
        String query = "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX dbpr: <http://dbpedia.org/resource/> SELECT * WHERE { dbpr:The_Troop rdf:type ?obj }";
        query = """
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX wd: <http://www.wikidata.org/entity/>
            SELECT (COUNT(DISTINCT (?occName))AS ?num_works) ?presName
            
            WHERE {
                
                ?president wdt:P39/wdt:P279* wd:Q248577 .
                Minus {
                    ?president wdt:P27 ?country .
                }
                ?president wdt:P106 ?occupation .
                Optional {
                    ?country <http://schema.org/name> ?countryName.
                }
                ?occupation <http://schema.org/name> ?occName.
                ?president <http://schema.org/name> ?presName.
                
                FILTER ( ( !REGEX(?occName, ".*[Pp]resid.*") ) )
                
            }
            GROUP BY ?work ?presName
            ORDER BY DESC (?num_works)
            """;
            query="""
                PREFIX : <http://dbpedia.org/resource/> PREFIX foaf: <http://xmlns.com/foaf/0.1/> SELECT DISTINCT ?page WHERE { ?syn (<http://dbpedia.org/ontology/wikiPageDisambiguates>)* :Lyon . ?syn foaf:isPrimaryTopicOf ?page }
                """;
            query="""
                PREFIX : <http://dbpedia.org/resource/> PREFIX foaf: <http://xmlns.com/foaf/0.1/> SELECT DISTINCT ?page WHERE { ?syn <http://dbpedia.org/ontology/wikiPageDisambiguates>|<http://dbpedia.org/ontology/wikiPageDisambiguates> :Lyon . ?syn foaf:isPrimaryTopicOf ?page }
                    """;
                Utils u = new Utils();
            query ="PREFIX category: <http://dbpedia.org/resource/Category:> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> SELECT DISTINCT ?super ?preferredLabel WHERE { ?super (^skos:broader){0,5} category:American_New_Wave_musicians . ?super (^skos:broader){0,5} category:Oral_hygiene . ?super skos:prefLabel ?preferredLabel }";
        u.create_algebra_test(query);
    }

    public static void testQuery() {
        Utils u = new Utils();
        String query = """
                PREFIX dbpo: <http://dbpedia.org/ontology/> PREFIX owl: <http://www.w3.org/2002/07/owl#> PREFIX dbpr: <http://dbpedia.org/resource/> PREFIX foaf: <http://xmlns.com/foaf/0.1/> SELECT * WHERE { <http://dbpedia.org/resource/Iv%C3%A1n_Amaya> ?p ?o FILTER ( ( ( ( ( ( ?p != owl:sameAs ) && ( ?p != foaf:thumbnail ) ) && ( ?p != dbpo:thumbnail ) ) && ( ?p != dbpo:wikiPageExternalLink ) ) && ( ?p != foaf:depiction ) ) && ( ?p != foaf:homepage ) ) }
                    """;
        System.out.println(u.time_query_plan(query));
    }

    public static void printQueryGraph(String queryString){
        System.out.println(getQueryGraph(queryString));
    }

    public static void disableWarns(){
        
            try{
                System.setErr(new PrintStream("/dev/null"));
                }catch (FileNotFoundException e){
                }
    }
    
    public static String getQueryGraph(String queryString){
        Query q = QueryFactory.create(queryString);
        Op o = Algebra.compile(q);
        o = Algebra.optimize(o);
        QueryGraphVisitor visitor = new QueryGraphVisitor();
        o.visit(visitor);
        visitor.finalize_edges();
        return visitor.graph();
    }

    public static String debugGetQueryGraph(String queryString){
        long start; 
        long queryPars;
        long queryOpt;
        long queryGr;
        
        start = System.currentTimeMillis();
        Query q = QueryFactory.create(queryString);
        Op o = Algebra.compile(q);
        queryPars = System.currentTimeMillis();
        
        
        o = Algebra.optimize(o);
        queryOpt = System.currentTimeMillis();
        
        
        QueryGraphVisitor visitor = new QueryGraphVisitor();
        
        o.visit(visitor);
        visitor.finalize_edges();
        String ret = visitor.graph();
        queryGr = System.currentTimeMillis();
        
        System.out.println("Time after query plan: " + (queryPars-start));
        System.out.println("Time after query plan opt: " + (queryOpt-queryPars));
        System.out.println("Time after query graph: " + (queryGr-queryOpt));
        
        
        return ret;
    }
    

}
