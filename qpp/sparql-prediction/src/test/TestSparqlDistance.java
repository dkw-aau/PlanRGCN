package test;

import static org.junit.Assert.*;

import org.junit.Test;

import semanticweb.sparql.preprocess.SparqlDistance;


public class TestSparqlDistance {

	@Test
	public void hungeraniaDistance() throws Exception {
		//String q2= "/sparql?default-graph-uri=http%3A%2F%2Fdbpedia.org&output=json&query=PREFIX%20rdfs%3A%20%3Chttp%3A%2F%2Fwww.w3.org%2F2000%2F01%2Frdf-schema%23%3E%20SELECT%20*%20WHERE%20%7B%20%3Fcity%20a%20%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2FPlace%3E%3B%20rdfs%3Alabel%20%27Boulaide%27%40en.%20%20%3Fairport%20a%20%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2FAirport%3E.%20%7B%3Fairport%20%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2Fcity%3E%20%3Fcity%7D%20UNION%20%7B%3Fairport%20%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2Flocation%3E%20%3Fcity%7D%20UNION%20%7B%3Fairport%20%3Chttp%3A%2F%2Fdbpedia.org%2Fproperty%2FcityServed%3E%20%3Fcity.%7D%20UNION%20%7B%3Fairport%20%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2Fcity%3E%20%3Fcity.%20%7D%7B%3Fairport%20%3Chttp%3A%2F%2Fdbpedia.org%2Fproperty%2Fiata%3E%20%3Fiata.%7D%20UNION%20%20%7B%3Fairport%20%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2FiataLocationIdentifier%3E%20%3Fiata.%20%7D%20OPTIONAL%20%7B%20%3Fairport%20foaf%3Ahomepage%20%3Fairport_home.%20%7D%20OPTIONAL%20%7B%20%3Fairport%20rdfs%3Alabel%20%3Fname.%20%7D%20OPTIONAL%20%7B%20%3Fairport%20%3Chttp%3A%2F%2Fdbpedia.org%2Fproperty%2Fnativename%3E%20%3Fairport_name.%7D%20FILTER%20%28%20%21bound%28%3Fname%29%20%7C%7C%20langMatches%28%20lang%28%3Fname%29%2C%20%27de%27%29%20%29%7D";
		//String q1 = "/sparql?query=PREFIX++dc%3A+++%3Chttp%3A%2F%2Fpurl.org%2Fdc%2Felements%2F1.1%2F%3E%0APREFIX++dbpedia2%3A+%3Chttp%3A%2F%2Fdbpedia.org%2Fproperty%2F%3E%0APREFIX++dbpedia-owl%3A+%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2F%3E%0APREFIX++geo%3A++%3Chttp%3A%2F%2Fwww.w3.org%2F2003%2F01%2Fgeo%2Fwgs84_pos%23%3E%0APREFIX++foaf%3A+%3Chttp%3A%2F%2Fxmlns.com%2Ffoaf%2F0.1%2F%3E%0APREFIX++yago%3A+%3Chttp%3A%2F%2Fmpii.de%2Fyago%2Fresource%2F%3E%0APREFIX++georss%3A+%3Chttp%3A%2F%2Fwww.georss.org%2Fgeorss%2F%3E%0APREFIX++geonames%3A+%3Chttp%3A%2F%2Fwww.geonames.org%2Fontology%23%3E%0APREFIX++umbel-ac%3A+%3Chttp%3A%2F%2Fumbel.org%2Fumbel%2Fac%2F%3E%0APREFIX++units%3A+%3Chttp%3A%2F%2Fdbpedia.org%2Funits%2F%3E%0APREFIX++dcterms%3A+%3Chttp%3A%2F%2Fpurl.org%2Fdc%2Fterms%2F%3E%0APREFIX++gn%3A+++%3Chttp%3A%2F%2Fxldb.di.fc.ul.pt%2Fxldb%2Fpublications%2F2009%2F10%2Fgeo-net%23%3E%0APREFIX++rdfs%3A+%3Chttp%3A%2F%2Fwww.w3.org%2F2000%2F01%2Frdf-schema%23%3E%0APREFIX++gnpt%3A+%3Chttp%3A%2F%2Fxldb.di.fc.ul.pt%2Fxldb%2Fpublications%2F2009%2F10%2Fgeo-net-pt%23%3E%0APREFIX++umbel-sc%3A+%3Chttp%3A%2F%2Fumbel.org%2Fumbel%2Fsc%2F%3E%0APREFIX++xsd%3A++%3Chttp%3A%2F%2Fwww.w3.org%2F2001%2FXMLSchema%23%3E%0APREFIX++owl%3A++%3Chttp%3A%2F%2Fwww.w3.org%2F2002%2F07%2Fowl%23%3E%0APREFIX++dbpedia%3A+%3Chttp%3A%2F%2Fdbpedia.org%2F%3E%0APREFIX++gnpt02%3A+%3Chttp%3A%2F%2Fxldb.di.fc.ul.pt%2Fxldb%2Fpublications%2F2009%2F10%2Fgeo-net-pt-02%23%3E%0APREFIX++rdf%3A++%3Chttp%3A%2F%2Fwww.w3.org%2F1999%2F02%2F22-rdf-syntax-ns%23%3E%0APREFIX++wikicompany%3A+%3Chttp%3A%2F%2Fdbpedia.openlinksw.com%2Fwikicompany%2F%3E%0APREFIX++skos%3A+%3Chttp%3A%2F%2Fwww.w3.org%2F2004%2F02%2Fskos%2Fcore%23%3E%0APREFIX++opencyc%3A+%3Chttp%3A%2F%2Fsw.opencyc.org%2F2008%2F06%2F10%2Fconcept%2F%3E%0A%0ASELECT++%3Fy+%3Fz%0AWHERE%0A++%7B+%3Chttp%3A%2F%2Fdbpedia.org%2Fresource%2FRose%3E+%3Fy+%3Fz+%7D%0A&default-graph-uri=http%3A%2F%2Fdbpedia.org";
		String q2 = "/sparql/?query=SELECT+%3Fabstract+WHERE+{+%3Chttp%3A%2F%2Fdbpedia.org%2Fresource%2FNational_Basketball_Association%3E+%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2Fabstract%3E+%3Fabstract.+FILTER+langMatches(lang(%3Fabstract)%2C+%27en%27)+}&format=json";
		String q1="/sparql/?query=SELECT+%3Fabstract+WHERE+{+%3Chttp%3A%2F%2Fdbpedia.org%2Fresource%2FHalf_dime%3E+%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2Fabstract%3E+%3Fabstract.+FILTER+langMatches(lang(%3Fabstract)%2C+%27en%27)+}&format=json";

		double d = SparqlDistance.dbpediaQueryGraphHumgarainDistance(q1, q2);
		System.out.println(d);
		
		String s = "b210140611b89b8a24f73170d8764d2e - - [10/Aug/2012 04:00:00 +0200] \"/sparql?query=PREFIX+dbp%3A+%3Chttp%3A%2F%2Fdbpedia.org%2Fresource%2F%3E%0A++++SELECT+%3Fdepiction%0A%09WHERE+%7B+%0A%09++dbp%3ACancer+foaf%3Adepiction+%3Fdepiction+.+%0A%09%7D&format=json\" 200 512 \"-\" \"-\"";
		String[] ss = s.split(" ");
		
		for (String sss:ss) {
			System.out.println(sss);
		}
		System.out.println("The query:"+ ss[6].substring(1, ss[6].length()-1));
	}

}
