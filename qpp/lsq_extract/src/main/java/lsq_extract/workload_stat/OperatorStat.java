package lsq_extract.workload_stat;



import java.util.ArrayList;

import org.apache.jena.query.ARQ;
import org.apache.jena.query.QueryParseException;
import org.apache.jena.sparql.expr.ExprEvalException;

import lsq_extract.util.Query;
import semanticweb.sparql.preprocess.AlgebraFeatureExtractor;

public class OperatorStat {
    ArrayList<Query> queries;
    static String[] header = new String[]{
        "triple", "bgp", "join", "leftjoin", "union", "filter", "graph", "extend", "minus", "path*",
        "pathN*", "path+", "pathN+", "path?", "notoneof", "tolist", "order", "project", "distinct", "reduced",
        "multi", "top", "group", "assign", "sequence", "slice", "treesize"};
    AlgebraFeatureExtractor fe;
    
    public OperatorStat(ArrayList<Query> qs){
        this.queries= qs;
        ARQ.init();
        fe = new AlgebraFeatureExtractor(header);
    }
    
    public ArrayList<Query> addOperatorStats(){
        int count = 1;
        int skipped = 0;
        int featureCountLessSkipped = 0;
        ArrayList<Query> parsedQueries = new ArrayList<>();
        for( Query i : queries){
            //System.out.println("Query "+count +" of " +queries.size());
            double[] features = null;
            try{
            features = fe.extractFeatures(i.text);
            }catch(QueryParseException e){
                System.out.println("ParseError: Did not work for: "+i.id);
                skipped++;
                continue;
            }catch(ExprEvalException e){
                System.out.println("Did not work for: "+i.id);
                skipped++;
                continue;
            }
            if (features == null){
                throw new RuntimeException("Should not be null");
            }
            for(int x = 0; x<header.length;x++){
                if(features[x] <0){
                    featureCountLessSkipped++;
                    continue;
                    //throw new RuntimeException("Feature Count less than 0 for Query"+ count+ " for Feature " +header[x]+" with value" + features[x]);
                }
                i.addOp(header[x], features[x]);
            }
            parsedQueries.add(i);
            count++;
        }
        System.out.println("Skiped " +(skipped+featureCountLessSkipped)+" queries");
        System.out.println("Of which "+featureCountLessSkipped+" had negative feature count");
        return parsedQueries;
    }
}
