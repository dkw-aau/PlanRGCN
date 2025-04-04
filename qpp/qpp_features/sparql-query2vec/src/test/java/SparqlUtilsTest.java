import org.apache.jena.datatypes.xsd.impl.XSDBaseNumericType;
import org.apache.jena.graph.Node;
import org.apache.jena.graph.Triple;
import org.apache.jena.query.*;
import org.apache.jena.sparql.algebra.Algebra;
import org.apache.jena.sparql.algebra.Op;
import org.apache.jena.sparql.core.TriplePath;
import org.apache.jena.sparql.syntax.*;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import semanticweb.sparql.Operator;
import semanticweb.sparql.QDistanceHungarian;
import semanticweb.sparql.SparqlUtils;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

@Disabled
public class SparqlUtilsTest {
    @Disabled
    @Test
    public void testRetrieveTriples() {
        String s = "PREFIX b3s: <http://b3s.openlinksw.com/> PREFIX bif: <bif:> PREFIX category:    <http://dbpedia.org/resource/Category:> PREFIX dawgt:   <http://www.w3.org/2001/sw/DataAccess/tests/test-dawg#> PREFIX dbpedia: <http://dbpedia.org/resource/> PREFIX dbpedia-owl: <http://dbpedia.org/ontology/> PREFIX dbpprop: <http://dbpedia.org/property/> PREFIX dc:  <http://purl.org/dc/elements/1.1/> PREFIX dcterms: <http://purl.org/dc/terms/> PREFIX fn:  <http://www.w3.org/2005/xpath-functions/#> PREFIX foaf:    <http://xmlns.com/foaf/0.1/> PREFIX freebase:    <http://rdf.freebase.com/ns/> PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#> PREFIX geonames:    <http://www.geonames.org/ontology#> PREFIX go:  <http://purl.org/obo/owl/GO#> PREFIX gr:  <http://purl.org/goodrelations/v1#> PREFIX grs: <http://www.georss.org/georss/> PREFIX lgv: <http://linkedgeodata.org/ontology/> PREFIX lod: <http://lod.openlinksw.com/> PREFIX math:    <http://www.w3.org/2000/10/swap/math#> PREFIX mesh:    <http://purl.org/commons/record/mesh/> PREFIX mf:  <http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#> PREFIX nci: <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#> PREFIX obo: <http://www.geneontology.org/formats/oboInOwl#> PREFIX opencyc: <http://sw.opencyc.org/2008/06/10/concept/> PREFIX owl: <http://www.w3.org/2002/07/owl#> PREFIX product: <http://www.buy.com/rss/module/productV2/> PREFIX protseq: <http://purl.org/science/protein/bysequence/> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfa:    <http://www.w3.org/ns/rdfa#> PREFIX rdfdf:   <http://www.openlinksw.com/virtrdf-data-formats#> PREFIX rdfs:    <http://www.w3.org/2000/01/rdf-schema#> PREFIX rev: <http://purl.org/stuff/rev#> PREFIX sc:  <http://purl.org/science/owl/sciencecommons/> PREFIX scovo:   <http://purl.org/NET/scovo#> PREFIX sd:  <http://www.w3.org/ns/sparql-service-description#> PREFIX sioc:    <http://rdfs.org/sioc/ns#> PREFIX skos:    <http://www.w3.org/2004/02/skos/core#> PREFIX sql: <sql:> PREFIX umbel-ac:    <http://umbel.org/umbel/ac/> PREFIX umbel-sc:    <http://umbel.org/umbel/sc/> PREFIX units:   <http://dbpedia.org/units/> PREFIX usc: <http://www.rdfabout.com/rdf/schema/uscensus/details/100pct/> PREFIX vcard:   <http://www.w3.org/2001/vcard-rdf/3.0#> PREFIX vcard2006:   <http://www.w3.org/2006/vcard/ns#> PREFIX virtcxml:    <http://www.openlinksw.com/schemas/virtcxml#> PREFIX virtrdf: <http://www.openlinksw.com/schemas/virtrdf#> PREFIX void:    <http://rdfs.org/ns/void#> PREFIX wdrs:    <http://www.w3.org/2007/05/powder-s#> PREFIX wikicompany: <http://dbpedia.openlinksw.com/wikicompany/> PREFIX xf:  <http://www.w3.org/2004/07/xpath-functions> PREFIX xml: <http://www.w3.org/XML/1998/namespace> PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> PREFIX xsl10:   <http://www.w3.org/XSL/Transform/1.0> PREFIX xsl1999: <http://www.w3.org/1999/XSL/Transform> PREFIX xslwd:   <http://www.w3.org/TR/WD-xsl> PREFIX yago:    <http://dbpedia.org/class/yago/> PREFIX yago-res:    <http://mpii.de/yago/resource/> PREFIX :     <http://example/> PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> \n" +
                "SELECT  ?long \n" +
                "WHERE\n" +
                "  {   { SELECT  ?long\n" +
                "        WHERE\n" +
                "          { <http://dbpedia.org/resource/Culebra,_Puerto_Rico> <http://www.w3.org/2003/01/geo/wgs84_pos#long> ?long }\n" +
                "      }\n" +
                "    UNION\n" +
                "      { SELECT  ?long\n" +
                "        WHERE\n" +
                "          { <http://dbpedia.org/resource/Culebra,_Puerto_Rico> <http://dbpedia.org/ontology/wikiPageRedirects> ?nuri .\n" +
                "            ?nuri <http://www.w3.org/2003/01/geo/wgs84_pos#long> ?long\n" +
                "          }\n" +
                "      }\n" +
                "  }\n";

        System.out.println("Original query: " + s);

        System.out.println("Extracted triples: ");
        Set<Triple> triples = SparqlUtils.retrieveTriples(s);

        for (Triple t : triples) {
            System.out.println(t.toString());
        }
    }
    @Disabled
    @Test
    public void testRetrieveTriplesPredicatesLSQ() {
        ArrayList<String[]> arrayVectores = SparqlUtils.getArrayFeaturesVector(
                "/home/daniel/Documentos/Web_Semantica/Work/Sparql2vec/lsq_extracted1000.csv",
                "/home/daniel/Documentos/ML/rhassan/graph-edit-distance/predicates_most_popular.csv",
                "/home/daniel/Documentos/Web_Semantica/Work/Sparql2vec/prefixes.txt",
                "datasetlsq.csv");
    }

    @Disabled
    @Test
    public void testReplacePrefixes() {
        String s = "PREFIX b3s: <http://b3s.openlinksw.com/> PREFIX bif: <bif:> PREFIX category:    <http://dbpedia.org/resource/Category:> PREFIX dawgt:   <http://www.w3.org/2001/sw/DataAccess/tests/test-dawg#> PREFIX dbpedia: <http://dbpedia.org/resource/> PREFIX dbpedia-owl: <http://dbpedia.org/ontology/> PREFIX dbpprop: <http://dbpedia.org/property/> PREFIX dc:  <http://purl.org/dc/elements/1.1/> PREFIX dcterms: <http://purl.org/dc/terms/> PREFIX fn:  <http://www.w3.org/2005/xpath-functions/#> PREFIX foaf:    <http://xmlns.com/foaf/0.1/> PREFIX freebase:    <http://rdf.freebase.com/ns/> PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#> PREFIX geonames:    <http://www.geonames.org/ontology#> PREFIX go:  <http://purl.org/obo/owl/GO#> PREFIX gr:  <http://purl.org/goodrelations/v1#> PREFIX grs: <http://www.georss.org/georss/> PREFIX lgv: <http://linkedgeodata.org/ontology/> PREFIX lod: <http://lod.openlinksw.com/> PREFIX math:    <http://www.w3.org/2000/10/swap/math#> PREFIX mesh:    <http://purl.org/commons/record/mesh/> PREFIX mf:  <http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#> PREFIX nci: <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#> PREFIX obo: <http://www.geneontology.org/formats/oboInOwl#> PREFIX opencyc: <http://sw.opencyc.org/2008/06/10/concept/> PREFIX owl: <http://www.w3.org/2002/07/owl#> PREFIX product: <http://www.buy.com/rss/module/productV2/> PREFIX protseq: <http://purl.org/science/protein/bysequence/> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfa:    <http://www.w3.org/ns/rdfa#> PREFIX rdfdf:   <http://www.openlinksw.com/virtrdf-data-formats#> PREFIX rdfs:    <http://www.w3.org/2000/01/rdf-schema#> PREFIX rev: <http://purl.org/stuff/rev#> PREFIX sc:  <http://purl.org/science/owl/sciencecommons/> PREFIX scovo:   <http://purl.org/NET/scovo#> PREFIX sd:  <http://www.w3.org/ns/sparql-service-description#> PREFIX sioc:    <http://rdfs.org/sioc/ns#> PREFIX skos:    <http://www.w3.org/2004/02/skos/core#> PREFIX sql: <sql:> PREFIX umbel-ac:    <http://umbel.org/umbel/ac/> PREFIX umbel-sc:    <http://umbel.org/umbel/sc/> PREFIX units:   <http://dbpedia.org/units/> PREFIX usc: <http://www.rdfabout.com/rdf/schema/uscensus/details/100pct/> PREFIX vcard:   <http://www.w3.org/2001/vcard-rdf/3.0#> PREFIX vcard2006:   <http://www.w3.org/2006/vcard/ns#> PREFIX virtcxml:    <http://www.openlinksw.com/schemas/virtcxml#> PREFIX virtrdf: <http://www.openlinksw.com/schemas/virtrdf#> PREFIX void:    <http://rdfs.org/ns/void#> PREFIX wdrs:    <http://www.w3.org/2007/05/powder-s#> PREFIX wikicompany: <http://dbpedia.openlinksw.com/wikicompany/> PREFIX xf:  <http://www.w3.org/2004/07/xpath-functions> PREFIX xml: <http://www.w3.org/XML/1998/namespace> PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> PREFIX xsl10:   <http://www.w3.org/XSL/Transform/1.0> PREFIX xsl1999: <http://www.w3.org/1999/XSL/Transform> PREFIX xslwd:   <http://www.w3.org/TR/WD-xsl> PREFIX yago:    <http://dbpedia.org/class/yago/> PREFIX yago-res:    <http://mpii.de/yago/resource/> PREFIX :     <http://example/> PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> \n SELECT  ?long\n" +
                "WHERE\n" +
                "  {   { SELECT  ?long\n" +
                "        WHERE\n" +
                "          { <http://dbpedia.org/resource/Culebra,_Puerto_Rico> geo:long ?long }\n" +
                "      }\n" +
                "    UNION\n" +
                "      { SELECT  ?long\n" +
                "        WHERE\n" +
                "          { <http://dbpedia.org/resource/Culebra,_Puerto_Rico> dbpedia-owl:wikiPageRedirects ?nuri .\n" +
                "            ?nuri geo:long ?long\n" +
                "          }\n" +
                "      }\n" +
                "  }";

        System.out.println("Original query: " + s);

        System.out.println("Extracted triples: ");
        String query = SparqlUtils.replacePrefixes(s);


        System.out.println(query);
    }

    @Disabled
    @Test
    public void testFixVariables() {
        String s = "PREFIX b3s: <http://b3s.openlinksw.com/> PREFIX bif: <bif:> PREFIX category:    <http://dbpedia.org/resource/Category:> PREFIX dawgt:   <http://www.w3.org/2001/sw/DataAccess/tests/test-dawg#> PREFIX dbpedia: <http://dbpedia.org/resource/> PREFIX dbpedia-owl: <http://dbpedia.org/ontology/> PREFIX dbpprop: <http://dbpedia.org/property/> PREFIX dc:  <http://purl.org/dc/elements/1.1/> PREFIX dcterms: <http://purl.org/dc/terms/> PREFIX fn:  <http://www.w3.org/2005/xpath-functions/#> PREFIX foaf:    <http://xmlns.com/foaf/0.1/> PREFIX freebase:    <http://rdf.freebase.com/ns/> PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#> PREFIX geonames:    <http://www.geonames.org/ontology#> PREFIX go:  <http://purl.org/obo/owl/GO#> PREFIX gr:  <http://purl.org/goodrelations/v1#> PREFIX grs: <http://www.georss.org/georss/> PREFIX lgv: <http://linkedgeodata.org/ontology/> PREFIX lod: <http://lod.openlinksw.com/> PREFIX math:    <http://www.w3.org/2000/10/swap/math#> PREFIX mesh:    <http://purl.org/commons/record/mesh/> PREFIX mf:  <http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#> PREFIX nci: <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#> PREFIX obo: <http://www.geneontology.org/formats/oboInOwl#> PREFIX opencyc: <http://sw.opencyc.org/2008/06/10/concept/> PREFIX owl: <http://www.w3.org/2002/07/owl#> PREFIX product: <http://www.buy.com/rss/module/productV2/> PREFIX protseq: <http://purl.org/science/protein/bysequence/> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfa:    <http://www.w3.org/ns/rdfa#> PREFIX rdfdf:   <http://www.openlinksw.com/virtrdf-data-formats#> PREFIX rdfs:    <http://www.w3.org/2000/01/rdf-schema#> PREFIX rev: <http://purl.org/stuff/rev#> PREFIX sc:  <http://purl.org/science/owl/sciencecommons/> PREFIX scovo:   <http://purl.org/NET/scovo#> PREFIX sd:  <http://www.w3.org/ns/sparql-service-description#> PREFIX sioc:    <http://rdfs.org/sioc/ns#> PREFIX skos:    <http://www.w3.org/2004/02/skos/core#> PREFIX sql: <sql:> PREFIX umbel-ac:    <http://umbel.org/umbel/ac/> PREFIX umbel-sc:    <http://umbel.org/umbel/sc/> PREFIX units:   <http://dbpedia.org/units/> PREFIX usc: <http://www.rdfabout.com/rdf/schema/uscensus/details/100pct/> PREFIX vcard:   <http://www.w3.org/2001/vcard-rdf/3.0#> PREFIX vcard2006:   <http://www.w3.org/2006/vcard/ns#> PREFIX virtcxml:    <http://www.openlinksw.com/schemas/virtcxml#> PREFIX virtrdf: <http://www.openlinksw.com/schemas/virtrdf#> PREFIX void:    <http://rdfs.org/ns/void#> PREFIX wdrs:    <http://www.w3.org/2007/05/powder-s#> PREFIX wikicompany: <http://dbpedia.openlinksw.com/wikicompany/> PREFIX xf:  <http://www.w3.org/2004/07/xpath-functions> PREFIX xml: <http://www.w3.org/XML/1998/namespace> PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> PREFIX xsl10:   <http://www.w3.org/XSL/Transform/1.0> PREFIX xsl1999: <http://www.w3.org/1999/XSL/Transform> PREFIX xslwd:   <http://www.w3.org/TR/WD-xsl> PREFIX yago:    <http://dbpedia.org/class/yago/> PREFIX yago-res:    <http://mpii.de/yago/resource/> PREFIX :     <http://example/> PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> \n SELECT  ?long\n" +
                "WHERE\n" +
                "  {   { SELECT  ?long\n" +
                "        WHERE\n" +
                "          { <http://dbpedia.org/resource/Culebra,_Puerto_Rico> geo:long ?long }\n" +
                "      }\n" +
                "    UNION\n" +
                "      { SELECT  ?long\n" +
                "        WHERE\n" +
                "          { <http://dbpedia.org/resource/Culebra,_Puerto_Rico> dbpedia-owl:wikiPageRedirects ?nuri .\n" +
                "            ?nuri geo:long ?long\n" +
                "          }\n" +
                "      }\n" +
                "  }";

        System.out.println("Original query: " + s);

        System.out.println("Extracted triples: ");
        String query = SparqlUtils.fixVariables(s);
        System.out.println(query);
    }

    @Disabled
    @Test
    public void testGetQueryAsTokens() {
        String s = "PREFIX b3s: <http://b3s.openlinksw.com/> PREFIX bif: <bif:> PREFIX category:    <http://dbpedia.org/resource/Category:> PREFIX dawgt:   <http://www.w3.org/2001/sw/DataAccess/tests/test-dawg#> PREFIX dbpedia: <http://dbpedia.org/resource/> PREFIX dbpedia-owl: <http://dbpedia.org/ontology/> PREFIX dbpprop: <http://dbpedia.org/property/> PREFIX dc:  <http://purl.org/dc/elements/1.1/> PREFIX dcterms: <http://purl.org/dc/terms/> PREFIX fn:  <http://www.w3.org/2005/xpath-functions/#> PREFIX foaf:    <http://xmlns.com/foaf/0.1/> PREFIX freebase:    <http://rdf.freebase.com/ns/> PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#> PREFIX geonames:    <http://www.geonames.org/ontology#> PREFIX go:  <http://purl.org/obo/owl/GO#> PREFIX gr:  <http://purl.org/goodrelations/v1#> PREFIX grs: <http://www.georss.org/georss/> PREFIX lgv: <http://linkedgeodata.org/ontology/> PREFIX lod: <http://lod.openlinksw.com/> PREFIX math:    <http://www.w3.org/2000/10/swap/math#> PREFIX mesh:    <http://purl.org/commons/record/mesh/> PREFIX mf:  <http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#> PREFIX nci: <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#> PREFIX obo: <http://www.geneontology.org/formats/oboInOwl#> PREFIX opencyc: <http://sw.opencyc.org/2008/06/10/concept/> PREFIX owl: <http://www.w3.org/2002/07/owl#> PREFIX product: <http://www.buy.com/rss/module/productV2/> PREFIX protseq: <http://purl.org/science/protein/bysequence/> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfa:    <http://www.w3.org/ns/rdfa#> PREFIX rdfdf:   <http://www.openlinksw.com/virtrdf-data-formats#> PREFIX rdfs:    <http://www.w3.org/2000/01/rdf-schema#> PREFIX rev: <http://purl.org/stuff/rev#> PREFIX sc:  <http://purl.org/science/owl/sciencecommons/> PREFIX scovo:   <http://purl.org/NET/scovo#> PREFIX sd:  <http://www.w3.org/ns/sparql-service-description#> PREFIX sioc:    <http://rdfs.org/sioc/ns#> PREFIX skos:    <http://www.w3.org/2004/02/skos/core#> PREFIX sql: <sql:> PREFIX umbel-ac:    <http://umbel.org/umbel/ac/> PREFIX umbel-sc:    <http://umbel.org/umbel/sc/> PREFIX units:   <http://dbpedia.org/units/> PREFIX usc: <http://www.rdfabout.com/rdf/schema/uscensus/details/100pct/> PREFIX vcard:   <http://www.w3.org/2001/vcard-rdf/3.0#> PREFIX vcard2006:   <http://www.w3.org/2006/vcard/ns#> PREFIX virtcxml:    <http://www.openlinksw.com/schemas/virtcxml#> PREFIX virtrdf: <http://www.openlinksw.com/schemas/virtrdf#> PREFIX void:    <http://rdfs.org/ns/void#> PREFIX wdrs:    <http://www.w3.org/2007/05/powder-s#> PREFIX wikicompany: <http://dbpedia.openlinksw.com/wikicompany/> PREFIX xf:  <http://www.w3.org/2004/07/xpath-functions> PREFIX xml: <http://www.w3.org/XML/1998/namespace> PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> PREFIX xsl10:   <http://www.w3.org/XSL/Transform/1.0> PREFIX xsl1999: <http://www.w3.org/1999/XSL/Transform> PREFIX xslwd:   <http://www.w3.org/TR/WD-xsl> PREFIX yago:    <http://dbpedia.org/class/yago/> PREFIX yago-res:    <http://mpii.de/yago/resource/> PREFIX :     <http://example/> PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> \n SELECT  ?long\n" +
                "WHERE\n" +
                "  {   { SELECT  ?long\n" +
                "        WHERE\n" +
                "          { <http://dbpedia.org/resource/Culebra,_Puerto_Rico> geo:long ?long }\n" +
                "      }\n" +
                "    UNION\n" +
                "      { SELECT  ?long\n" +
                "        WHERE\n" +
                "          { <http://dbpedia.org/resource/Culebra,_Puerto_Rico> dbpedia-owl:wikiPageRedirects ?nuri .\n" +
                "            ?nuri geo:long ?long\n" +
                "          }\n" +
                "      }\n" +
                "  }";

        System.out.println("Original query: " + s);

        System.out.println("Extracted triples: ");
        String[] query = SparqlUtils.getQueryAsTokens(s);
        for (int i = 0; i < query.length; i++) {
            System.out.println(query[i]);
        }
        System.out.println(query.length);
    }

    @Disabled
    @Test
    public void testSparql() throws Exception {
        String q1 = "PREFIX foaf:    <http://xmlns.com/foaf/0.1/> SELECT ?name ?email WHERE {  ?x foaf:knows ?y . ?y foaf:name ?name . ?a ?b <http://wimmics.inria.fr/kolflow/qp#tt>.  OPTIONAL { ?y foaf:mbox ?email }  }";
        String q2 = "PREFIX foaf:    <http://xmlns.com/foaf/0.1/> SELECT ?name ?email WHERE {  ?x foaf:knows ?y . ?y foaf:name ?name . ?a ?b <http://wimmics.inria.fr/kolflow/qp#tt> }";
        System.out.println(QDistanceHungarian.distance(q1, q2));
    }
    @Disabled
    @Test
    public void testSparql2() throws Exception {
        File file = new File("/home/daniel/Documentos/Web_Semantica/Work/Sparql2vec/6000/x_query.txt");


        BufferedReader br = new BufferedReader(new FileReader(file));
        ArrayList<String[]> arrayList = new ArrayList<String[]>();
        String st;
        int line = 0;
        while ((st = br.readLine()) != null)
            try {
                System.out.println(line);
                line++;
                st = java.net.URLDecoder.decode(st, StandardCharsets.UTF_8.name());
                st = st.substring(st.toLowerCase().indexOf("query=") + 6);
                arrayList.add(SparqlUtils.getQueryAsTokens(st));
            } catch (UnsupportedEncodingException e) {
                // not going to happen - value came from JDK's own StandardCharsets
            }
        SparqlUtils.produceCsvArray(arrayList, "my.txt");

    }
    @Disabled
    @Test
    public void testGetPropsAndObjectCount() throws Exception {
        SparqlUtils.getPropsAndObjectCount();
    }
    @Disabled
    @Test
    public void testGetQueryReadyForExecution() throws Exception {
        String s = "PREFIX b3s: <http://b3s.openlinksw.com/> PREFIX bif: <bif:> PREFIX category:    <http://dbpedia.org/resource/Category:> PREFIX dawgt:   <http://www.w3.org/2001/sw/DataAccess/tests/test-dawg#> PREFIX dbpedia: <http://dbpedia.org/resource/> PREFIX dbpedia-owl: <http://dbpedia.org/ontology/> PREFIX dbpprop: <http://dbpedia.org/property/> PREFIX dc:  <http://purl.org/dc/elements/1.1/> PREFIX dcterms: <http://purl.org/dc/terms/> PREFIX fn:  <http://www.w3.org/2005/xpath-functions/#> PREFIX foaf:    <http://xmlns.com/foaf/0.1/> PREFIX freebase:    <http://rdf.freebase.com/ns/> PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#> PREFIX geonames:    <http://www.geonames.org/ontology#> PREFIX go:  <http://purl.org/obo/owl/GO#> PREFIX gr:  <http://purl.org/goodrelations/v1#> PREFIX grs: <http://www.georss.org/georss/> PREFIX lgv: <http://linkedgeodata.org/ontology/> PREFIX lod: <http://lod.openlinksw.com/> PREFIX math:    <http://www.w3.org/2000/10/swap/math#> PREFIX mesh:    <http://purl.org/commons/record/mesh/> PREFIX mf:  <http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#> PREFIX nci: <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#> PREFIX obo: <http://www.geneontology.org/formats/oboInOwl#> PREFIX opencyc: <http://sw.opencyc.org/2008/06/10/concept/> PREFIX owl: <http://www.w3.org/2002/07/owl#> PREFIX product: <http://www.buy.com/rss/module/productV2/> PREFIX protseq: <http://purl.org/science/protein/bysequence/> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfa:    <http://www.w3.org/ns/rdfa#> PREFIX rdfdf:   <http://www.openlinksw.com/virtrdf-data-formats#> PREFIX rdfs:    <http://www.w3.org/2000/01/rdf-schema#> PREFIX rev: <http://purl.org/stuff/rev#> PREFIX sc:  <http://purl.org/science/owl/sciencecommons/> PREFIX scovo:   <http://purl.org/NET/scovo#> PREFIX sd:  <http://www.w3.org/ns/sparql-service-description#> PREFIX sioc:    <http://rdfs.org/sioc/ns#> PREFIX skos:    <http://www.w3.org/2004/02/skos/core#> PREFIX sql: <sql:> PREFIX umbel-ac:    <http://umbel.org/umbel/ac/> PREFIX umbel-sc:    <http://umbel.org/umbel/sc/> PREFIX units:   <http://dbpedia.org/units/> PREFIX usc: <http://www.rdfabout.com/rdf/schema/uscensus/details/100pct/> PREFIX vcard:   <http://www.w3.org/2001/vcard-rdf/3.0#> PREFIX vcard2006:   <http://www.w3.org/2006/vcard/ns#> PREFIX virtcxml:    <http://www.openlinksw.com/schemas/virtcxml#> PREFIX virtrdf: <http://www.openlinksw.com/schemas/virtrdf#> PREFIX void:    <http://rdfs.org/ns/void#> PREFIX wdrs:    <http://www.w3.org/2007/05/powder-s#> PREFIX wikicompany: <http://dbpedia.openlinksw.com/wikicompany/> PREFIX xf:  <http://www.w3.org/2004/07/xpath-functions> PREFIX xml: <http://www.w3.org/XML/1998/namespace> PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> PREFIX xsl10:   <http://www.w3.org/XSL/Transform/1.0> PREFIX xsl1999: <http://www.w3.org/1999/XSL/Transform> PREFIX xslwd:   <http://www.w3.org/TR/WD-xsl> PREFIX yago:    <http://dbpedia.org/class/yago/> PREFIX yago-res:    <http://mpii.de/yago/resource/> PREFIX :     <http://example/> PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> \n SELECT  ?long\n" +
                "WHERE\n" +
                "  {   { SELECT  ?long\n" +
                "        WHERE\n" +
                "          { <http://dbpedia.org/resource/Culebra,_Puerto_Rico> geo:long ?long }\n" +
                "      }\n" +
                "    UNION\n" +
                "      { SELECT  ?long\n" +
                "        WHERE\n" +
                "          { <http://dbpedia.org/resource/Culebra,_Puerto_Rico> dbpedia-owl:wikiPageRedirects ?nuri .\n" +
                "            ?nuri geo:long ?long\n" +
                "          }\n" +
                "      }\n" +
                "  }";

        System.out.println("Original query: " + s);

        System.out.println("Cleaned query: ");

        System.out.println(SparqlUtils.getQueryReadyForExecution(s));
    }
    @Disabled
    @Test
    public void testFeatureExtractDeepSet() {
        //String s = "SELECT * { ?s <http://purl.org/dc/elements/1.1/title> ?o1." +
        //		"?s <http://purl.org/dc/elements/1.1/description> ?o2. }";
        ArrayList<String> tablesOrder = new ArrayList<>();
        ArrayList<String> joinsOrder = new ArrayList<>();
        ArrayList<String> predicatesOrder = new ArrayList<>();
        ArrayList<String> predicatesUrisOrder = new ArrayList<>();

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
        for (int i = 0; i < queries.length; i++) {
            Query query = QueryFactory.create(queries[i]);
            System.out.println(query);

            // Generate algebra
            Op op = Algebra.compile(query);
            op = Algebra.optimize(op);
            System.out.println("AL: " + op);

            Element e = query.getQueryPattern();

            System.out.println("pattern:" + e);

            System.out.println("walk");
            ArrayList<String> queryTables = new ArrayList<>();
            ArrayList<String> queryVariables = new ArrayList<>();
            ArrayList<String> queryJoins = new ArrayList<>();
            ArrayList<HashMap<String, Object>> queryPredicates = new ArrayList<>();
            ArrayList<HashMap<String, Object>> queryPredicatesUris = new ArrayList<>();
            /**
             * Procedimiento:
             * Por cada triple de la forma ?a pred1 ?b:
             *  registro una tabla(tableId del pred1).
             *  Check if ?a exist in Object list
             */
            HashMap<String, ElementPathBlock> tpf = new HashMap<>();
//        AlgebraFeatureExtractor.getFeaturesDeepSet(op);
//        // This will walk through all parts of the query
            ElementWalker.walk(e,
                    // For each element...
                    new ElementVisitorBase() {
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
                            tpf.put(el.toString(), el);
                        }

                        public void visit(ElementTriplesBlock el) {
                            // ...go through all the triples...
                            Iterator<Triple> triples = el.patternElts();
                            while (triples.hasNext()) {
                                // ...and grab the subject
                                //subjects.add(triples.next().getSubject());
                                Triple t = triples.next();
                                System.out.println(t.toString());
                            }
                        }
                    }
            );
            ElementPathBlock[] elements = (ElementPathBlock[]) tpf.values().toArray();

            // Loop over elements without optional triples.
            for (ElementPathBlock el : elements) {
                Iterator<TriplePath> triples = el.patternElts();

                while (triples.hasNext()) {
                    // ...and grab the subject
                    TriplePath t = triples.next();
                    Node subject = t.getSubject();
                    Node object = t.getObject();
                    Node predicate = t.getPredicate();

                    if (subject.isVariable() && predicate.isURI() && object.isVariable()) {//if not int table list add to.
                        ArrayList[] res = this.processVarPredVar(queryTables,queryVariables,queryJoins,subject,predicate,object);
                        queryTables = res[0];
                        queryVariables = res[1];
                        queryJoins = res[2];
                    }
                    /**
                     * Logic for subject var, predicate uri, object int literal like (Var1.foaf:age, 29 )
                     */
                    else if (subject.isVariable()
                            && predicate.isURI()
                            && object.getLiteralDatatype().getClass() == XSDBaseNumericType.class) {//if not int table list add to.
//                                else if (subject.isVariable() && predicate.isURI() && object.isLiteral()) {//if not int table list add to.
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
                     * Logic for subject var, predicate uri, object uri like (Var1.rdf:type, foaf:Person)
                     */
                    else if (subject.isVariable() && predicate.isURI() && object.isURI()) {//if not int table list add to.
                        if (!queryTables.contains(predicate.getURI())) {
                            queryTables.add(predicate.getURI());
                        }
                        //add variables subject to list
                        if (!queryVariables.contains(subject.getName())) {
                            queryVariables.add(subject.getName());
                        }
                        HashMap<String, Object> pred = new HashMap<>();
                        pred.put("col", predicate.getURI());
                        pred.put("object", object.getURI());
                        queryPredicatesUris.add(pred);
                    }
                    // Todo Incorporate other cases...
                }
            }
            // Todo Codificate query and add to list.
            String a = "";
        }
    }


    private ArrayList[] processVarPredVar(
            ArrayList<String> queryTables,
            ArrayList<String> queryVariables,
            ArrayList<String> queryJoins,
            Node subject,
            Node predicate,
            Node object) {
        if (!queryTables.contains(predicate.getURI())) {
            queryTables.add(predicate.getURI());
        }
        //add variables subject to list
        if (!queryVariables.contains(subject.getName())) {
            queryVariables.add(subject.getName());
        }
        //add variables object to list
        if (!queryVariables.contains(object.getName())) {
            queryVariables.add(object.getName());
        }
        //add joins  var1_predicateURI_var2
        queryJoins.add(
                ""
                        .concat("v")
                        .concat(
                                String.valueOf(queryVariables.indexOf(subject.getName()))
                        )
                        .concat("_")
                        .concat(predicate.getURI())
                        .concat("_")
                        .concat("v")
                        .concat(
                                String.valueOf(queryVariables.indexOf(object.getName()))
                        )
        );
        return new ArrayList[]{queryTables, queryVariables, queryJoins};
    }
    @Disabled
    @Test
    public void extractData(){
        String qs =
                "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>" +
                "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " +
                "PREFIX lsqr: <http://lsq.aksw.org/res/>  " +
                "PREFIX lsqv: <http://lsq.aksw.org/vocab#>  " +
                "PREFIX sp: <http://spinrdf.org/sp#>  " +
                "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>  " +
                "PREFIX purl: <http://purl.org/dc/terms/>  \n" +
                "\n" +
                "SELECT  Distinct ?s (xsd:dateTime(?issued) as ?date) ?issued ?runTimeMs WHERE {  \n" +
                " ?s  lsqv:execution ?execution . \n" +
                "  ?execution purl:issued  ?issued  . \n" +
                "?s  lsqv:resultSize ?resultSize     .\n" +
                "\t  ?s  lsqv:runTimeMs ?runTimeMs     .\n" +
                "\t  ?s  rdf:type ?type     .\n" +
                "\t  ?s  sp:text ?query     .\n" +
                "      ?execution <http://lsq.aksw.org/vocab#agent> ?agent .\n" +
                "}\n" +
                "\n";
         String qs2 =
                "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>" +
                "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " +
                "PREFIX lsqr: <http://lsq.aksw.org/res/>  " +
                "PREFIX lsqv: <http://lsq.aksw.org/vocab#>  " +
                "PREFIX sp: <http://spinrdf.org/sp#>  " +
                "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>  " +
                "PREFIX purl: <http://purl.org/dc/terms/>  \n" +
                "\n" +
                "PREFIX id:   <http://dblp.rkbexplorer.com/id/>\n" +
                        "PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n" +
                        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n" +
                        "PREFIX akt:  <http://www.aktors.org/ontology/portal#>\n" +
                        "PREFIX owl:  <http://www.w3.org/2002/07/owl#>\n" +
                        "PREFIX akt:  <http://www.aktors.org/ontology/portal#>\n" +
                        "PREFIX akts: <http://www.aktors.org/ontology/support#>\n" +
                        "\n" +
                        "\n" +
                        "\n" +
                        "SELECT * WHERE { ?s rdfs:label ?o } LIMIT 10\n";

        Query query = QueryFactory.create(qs2) ;
        QueryExecution exec = QueryExecutionFactory.sparqlService("http://localhost:3332/dblp/sparql", query);

        ResultSet results = exec.execSelect();
//        try {
//            BufferedWriter prop_count = new BufferedWriter(new FileWriter("time_features_lsqdbpedia.csv"));
//            StringBuilder sb2 = new StringBuilder();
//            String separator = "ᶶ";
//            sb2.append("id");
//            sb2.append(separator);
//            sb2.append("date");
//            sb2.append(separator);
//            sb2.append("year");
//            sb2.append(separator);
//            sb2.append("month");
//            sb2.append(separator);
//            sb2.append("day");
//            sb2.append(separator);
//            sb2.append("time");
//            sb2.append(separator);
//            sb2.append("runtime");
//            sb2.append(LineSeparator.Unix);
//
            while (results.hasNext()) {
                QuerySolution a = results.next();
//                String id = String.valueOf(a.get("s"));
                System.out.println(a.toString());
            }
//            prop_count.write(sb2.toString());
//            prop_count.close();
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
    }
}