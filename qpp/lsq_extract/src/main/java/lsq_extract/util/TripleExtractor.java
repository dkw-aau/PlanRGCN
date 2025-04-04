package lsq_extract.util;


import java.util.LinkedList;
import org.apache.jena.query.ARQ;
import org.apache.jena.query.Query;
import org.apache.jena.query.QueryFactory;
import org.apache.jena.sparql.core.TriplePath;
import org.apache.jena.sparql.syntax.ElementPathBlock;
import org.apache.jena.sparql.syntax.ElementVisitorBase;
import org.apache.jena.sparql.syntax.ElementWalker;

import java.util.HashSet;
import java.util.Iterator;

public class TripleExtractor {
    LinkedList<HashSet<Triples>> tripleSet;

    public TripleExtractor() {
        ARQ.init();
        tripleSet = new LinkedList<HashSet<Triples>>();
    }

    public LinkedList<HashSet<Triples>> retrievesTriples(String queryString){
        Query q = QueryFactory.create(queryString);
        if (!q.isSelectType()){
            return new LinkedList<HashSet<Triples>>();
        }

        ElementWalker.walk(q.getQueryPattern(),
                new ElementVisitorBase() {
                    public void visit(ElementPathBlock el) {
                        Iterator<TriplePath> triples = el.patternElts();
                        HashSet<Triples> trplList = new HashSet<>();
                        while (triples.hasNext()) {
                            // ...and grab the subject
                            TriplePath temp = triples.next();
                            Triples t = new Triples(temp.getSubject().toString(),temp.getPredicate().toString(),temp.getObject().toString());
                            trplList.add(t);
                            //tripleSet.add(t);
                        }
                        tripleSet.add(trplList);
                    }
                }
        );
        return tripleSet;
    }

    public void clear(){
        this.tripleSet = new LinkedList<HashSet<Triples>>();
    }
}

