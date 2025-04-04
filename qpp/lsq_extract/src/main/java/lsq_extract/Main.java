

package lsq_extract;
import java.util.LinkedList;

import lsq_extract.lsq_execution.LSQExcutorDurationV2;
import lsq_extract.lsq_execution.LSQExecutor;
import lsq_extract.lsq_execution.LSQExecutorDurationFinal;
import lsq_extract.util.LSQDataReader;
import lsq_extract.util.LegalQueryChecker;
import lsq_extract.util.QueryReaderWAlgRunTime;
import lsq_extract.util.TripleExtractor;
import lsq_extract.util.Triples;
import lsq_extract.workload_stat.StatRunner;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;


public class Main {
    

    public static void main(String[] args){
        String commandChoices = "Choose between 1) extract-original 2) extract-improved (default) 3) legal-query-check 4) triples 5) final-interval";
        if( args.length == 0){
            System.out.println("Please write a command\n");
            System.out.println(commandChoices);
            System.exit(-1);
        }
        String task = args[0];
        try{
            switch (task) {
                //case "extract" -> extractLSQQueryLog(args[1]);
                //case "extract-original" -> new LSQExecutorBase().extractLSQQueryLog(args[1]);
                case "extract" -> new LSQExecutor(args[1], Integer.parseInt( args[3])).extractLSQQueryLog(args[2]);//1,464.143 s/24 min 37 s. for executing this.
                // arg 1: serviceURL, arg2: limit in each interval, arg 3: output path
                // java -jar lsq_extract-1.0-SNAPSHOT-jar-with-dependencies.jar  extract-interval http://172.21.233.23:8891/sparql/ 20000 /Users/abirammohanaraj/Documents/GitHub/dkw/qpp/data/lsqrun/dbpedia2016interval.csv
                //deprecated versions
                //case "extract-interval" -> new LSQExecutorDuration(args[1], Integer.parseInt(args[2])).extractQueries(args[3]); 
                //case "refined-interval" -> new LSQExecutorDuration(args[1], Integer.parseInt(args[2])).extractRefinedIntervalQueries(args[3]);
                case "extract-interval" -> new LSQExcutorDurationV2(args[1], Integer.parseInt(args[2])).extractQueries(args[3]); 
                case "refined-interval" -> new LSQExcutorDurationV2(args[1], Integer.parseInt(args[2])).extractRefinedIntervalQueries(args[3]);
                case "legal-query-check" -> processLegalQueries(args[1], args[2], args[3]);
                case "final-interval" -> new LSQExecutorDurationFinal(args[1], Integer.parseInt(args[2])).extractQueries(args[3]);
                case "triples" -> extractTriples(args[1]);
                case "workload-stat" -> new StatRunner(args[1]).RunStat(args[2]); //args 1: querylog path, args 2: output path.
                default -> System.out.println(commandChoices);
        }
        }catch(IOException e){}
    }



    public static void extractTriples(String query){
        query = query.replaceAll("§","\"");
        TripleExtractor t = new TripleExtractor();
        LinkedList<HashSet<Triples>> BGPs =  t.retrievesTriples(query);
        for (HashSet<Triples> triples: BGPs) {
            for (Triples i : triples) {
                System.out.println(i.toString());
            }
            System.out.println("§§");
        }
    }
    public static void processLegalQueries(String input_path, String legal_path, String illegal_path){
        LSQDataReader reader = new LSQDataReader(input_path);
        try {
            reader.readFile();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        reader.close();
        LegalQueryChecker checker = new LegalQueryChecker(legal_path, illegal_path);
        checker.legalQueryProcessor(reader);
    }
}