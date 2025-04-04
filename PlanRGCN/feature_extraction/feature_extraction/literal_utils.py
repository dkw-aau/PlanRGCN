from feature_extraction.predicates.pred_util import *
from unicodedata import normalize
import rdflib.plugins.sparql.parser as SPARQLparser
import urllib
from datetime import datetime


class LiteralFreqExtractor(ExtractorBase):
    def __init__(
        self,
        endpoint: Endpoint,
        output_dir: str,
        literal_file="literals.json",
        time_log="lit_freq_time.log",
        batch_size = 1000
    ) -> None:
        super().__init__(endpoint, output_dir, literal_file)
        self.time_log = f"{output_dir}/{time_log}"
        
        self.batch_size = batch_size
        self.literal_file = literal_file
        if os.path.exists(literal_file):
            literals = json.load(open(literal_file, "r"))
            self.literals = [x['o']['value'] for x in literals['results']['bindings'] ]
            self.literal_types = [x['o']['type'] for x in literals['results']['bindings'] ]
            # for backward compatibility (load_batches)
            self.predicates = self.literals
        
        self.batch_output_dir = f"{output_dir}/literals_stat/batches"
        os.system(f"mkdir -p {self.batch_output_dir}")
        # the
        # path to where the responses are saved with the features.
        self.batch_output_response_dir = f"{output_dir}/literals_stat/batches_response_stats"
        os.system(f"mkdir -p {self.batch_output_response_dir}")
        os.system(f"mkdir -p {self.batch_output_response_dir}/freq")
        os.system(f"mkdir -p {self.batch_output_response_dir}/pred_lits")

    def query_distinct_lits(self):
        query = LiteralStatQueries.extract_all_literals()
        start = time.time()
        res = self.endpoint.run_query(query)
        dur = time.time() - start
        res_fp = f"{self.literal_file}"
        json.dump(res, open(res_fp, "w"))
        with open(f"{self.output_dir}/literals_response_time.txt", 'w') as f:
            f.write(f"Literals extraction time: {dur}\n")
        print(f"Literals extracted!")
        
    def query_batches(self, batch_start=1, batch_end=2):
        if not hasattr(self, "batches"):
            self.load_batches()
        if batch_end == -1:
            batch_end = len(self.batches)
        save_path = self.batch_output_response_dir
        os.system(f"mkdir -p {save_path}")
        print(f"Literals Stats are saved to: {save_path}")
        print(f"Beginning extraction of batch {batch_start - 1} to {batch_end - 1}")
        f_time = open(self.time_log, 'a')
        for i, b in enumerate(self.batches[batch_start - 1 : batch_end - 1]):
            for query_generator, name in zip(
                [
                    LiteralStatQueries.freq_lits,
                ],
                ["freq"],
            ):
                res_fp = f"{save_path}/{name}/batch_{batch_start+i}.json"
                if os.path.exists(res_fp):
                    print(f"skipping {batch_start+i}!")
                    continue
                query = query_generator(b)
                try:
                    start = time.time()
                    res = self.endpoint.run_query(query)
                    dur = time.time()-start
                    json.dump(res, open(res_fp, "w"))
                    f_time.write(f"batch {batch_start+i}, {name}, {dur}\n")
                    f_time.flush()
                    print(f"batch {batch_start+i}/{len(self.batches)} extracted!")
                except TimeoutError:
                    save_long = f"{save_path}/timedout/batch_{batch_start+i}.json"
                    os.system(f"mkdir -p {save_path}/timedout")
                    print(f"batch {batch_start+i}/{len(self.batches)} timed out!")
                    with open(save_long, 'w') as tfp:
                        json.dump(b,tfp)
                except Exception:
                    print(f"Did not work for {batch_start+i}")
                    with open("/data/unprocessed_batches3.log","a") as f:
                        f.write(query)
                        f.write("\n\n\n\n")
        f_time.close()
        print("exiting after freq")
        exit()
        for i, b in enumerate(self.batches[batch_start - 1 : batch_end - 1]):
            for query_generator, name in zip(
                [
                    LiteralStatQueries.pred_lits,
                ],
                [ "pred_lits"],
            ):
                query = query_generator(b)
                res = self.endpoint.run_query(query)
                
                res_fp = f"{save_path}/{name}/batch_{batch_start+i}.json"
                json.dump(res, open(res_fp, "w"))
                print(f"batch {batch_start+i}/{len(self.batches)} extracted!")
    
    def query_batches_lit_specific(self, batch_start=1, batch_end=2):
        """Extract literal frequencies specific to literals

        Args:
            batch_start (int, optional): _description_. Defaults to 1.
            batch_end (int, optional): _description_. Defaults to 2.
        """
        if not hasattr(self, "batches"):
            self.load_batches()
        if batch_end == -1:
            batch_end = len(self.batches)
        save_path = self.batch_output_response_dir
        os.system(f"mkdir -p {save_path}")
        print(f"Literals Stats are saved to: {save_path}")
        print(f"Beginning extraction of batch {batch_start - 1} to {batch_end - 1}")
        f_time = open(self.time_log, 'a')
        for i, b in enumerate(self.batches[batch_start - 1 : batch_end - 1]):
            for query_generator, name in zip(
                [
                    LiteralStatQueries.freq_lit,
                ],
                ["freq"],
            ):
                res_fp = f"{save_path}/{name}/batch_{batch_start+i}"
                if os.path.exists(res_fp) or os.path.exists(f"{res_fp}.json"):
                    print(f"skipping {batch_start+i}!")
                    continue
                os.mkdir(res_fp)
                for idx_lit, lit in enumerate(b):
                    start = time.time()
                    try:
                        query = query_generator(lit)
                        if query is None:
                            continue
                        res = self.endpoint.run_query(query)
                        dur = time.time()-start
                        val = res['results']['bindings'][0]['literals']['value']
                        json.dump({lit[0]: val}, open(f"{res_fp}/{idx_lit}.json", "w"))
                        f_time.write(f"batch {batch_start+i}_{idx_lit}, {name}, {dur}\n")
                        f_time.flush()
                    except TimeoutError:
                        print(f"Did not work for {idx_lit,lit[0]}")
                        f_time.write(f"batch {batch_start+i}_{idx_lit}, {name}, -1, {lit[0]}\n")
                        f_time.flush()
                    except Exception:
                        print(f"Did not work for {batch_start+i}")
                        with open(f"/data/unprocessed_batches_{datetime.today().strftime('%Y_%m_%d')}.log","a") as f:
                            f.write(query)
                            f.write("\n\n\n\n")
        f_time.close()

class LiteralStatQueries:
    def extract_all_literals():
        return """
    SELECT distinct( ?o) {
        ?s ?p ?o.
        FILTER isLiteral(?o)
    }
    """
    def pred_str_gen(batch):
        pred_str = ""
        batch = LiteralStatQueries.retrieve_good_literals(batch)
        for p in batch:
            """if '"' in p[0]:
                p[0] = p[0].replace('"', '\\"').replace('\\\\', '\\')"""
            if ("Point" in p[0]):
                continue
            if p[1] == 'typed-literal':
                pred_str += f"({p[0]})"
            else:
                pred_str += f"(\"{p[0]}\")"
        return pred_str

    def retrieve_good_literals(batch, error_file = "/data/bad_literals.txt"):
        good_lits = []
        illegal_terms = []
        legal_terms = []
        for i in batch:
            if i[1] == "literal":
                query1 = f"""SELECT * WHERE {{ ?s ?p \"{i[0]}\"}}"""
            else:
                query1 = f"""SELECT * WHERE {{ ?s ?p {i[0]}}}"""
            try:
                SPARQLparser.parseQuery(query1)
            except Exception as e:
                illegal_terms.append(i)
            else:
                legal_terms.append(i)
        
        with open(error_file, 'a') as f:
            f.write(str(illegal_terms))
        
        return legal_terms
                

    def freq_lits(batch):
        """Generates the query for a batch (a list of predicates)"""
        ent_str = LiteralStatQueries.pred_str_gen(batch)
        return f"""SELECT ?e (COUNT( *) AS ?entities) WHERE {{
            VALUES (?e) {{ {ent_str}}}
        ?s ?p2 ?e .
        }}
        GROUP BY ?e
        """
        
    def freq_lit(lit):
        """Generates the query for a specific literals. The input is a list of literal value and whether it a typed literal"""
        if ("Point" in lit[0]):
            return
        if lit[1] == 'typed-literal':
            lit = f"{lit[0]}"
        else:
            if "\"" in lit[0]:
                lit = f"'{lit[0]}'"
            else:
                lit = f"\"{lit[0]}\""
                
        return f"""SELECT (COUNT( *) AS ?literals) WHERE {{
        ?s ?p2 {lit} .
        }}
        """

    def pred_lits(batch):
        ent_str = LiteralStatQueries.pred_str_gen(batch)
        """This returns the count of unique entities in both subject and object positions."""
        return f"""SELECT ?e (COUNT( ?p2) AS ?entities) WHERE {{
            VALUES (?e) {{ {ent_str}}}
            ?s ?p2 ?e .
        }}
        GROUP BY ?e
        """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Literal Stat Extractor",
        description="Extract Literal stats from SPARQL Endpoint",
        epilog="Text at the bottom of help",
    )

    parser.add_argument("task")
    parser.add_argument("-e", "--endpoint")
    parser.add_argument("--dir", "--output_dir")
    parser.add_argument("--lits_file")
    
    parser.add_argument("--batch_start")
    parser.add_argument("--batch_end")
    parser.add_argument("--timeout", type=int, default=-1)
    parser.add_argument("--time_log")

    args = parser.parse_args()

    if args.task == "distinct-literals":
        output_dir = f"{args.dir}"
        os.system(f"mkdir -p {output_dir}")
        endpoint = Endpoint(args.endpoint)
        os.system(f"mkdir -p {args.dir}")
        extractor = LiteralFreqExtractor(endpoint, output_dir, args.lits_file)
        extractor.query_distinct_lits()
        
    if args.task == "extract-lits-stat":
        output_dir = f"{args.dir}"
        os.system(f"mkdir -p {output_dir}")
        
        endpoint = Endpoint(args.endpoint)
        if args.timeout != -1:
            print(f"Timeout {args.timeout} set")
            endpoint.sparql.setTimeout(args.timeout)
        os.system(f"mkdir -p {args.dir}")
        extractor = LiteralFreqExtractor(endpoint, output_dir, args.lits_file, batch_size=500,  time_log=args.time_log)
        extractor.load_batches()
        extractor.query_batches(int(args.batch_start), int(args.batch_end))
    elif args.task == "extract-lits-statv2":
        output_dir = f"{args.dir}"
        os.system(f"mkdir -p {output_dir}")
        
        endpoint = Endpoint(args.endpoint)
        if args.timeout != -1:
            print(f"Timeout {args.timeout} set")
            endpoint.sparql.setTimeout(args.timeout)
        os.system(f"mkdir -p {args.dir}")
        extractor = LiteralFreqExtractor(endpoint, output_dir, args.lits_file, batch_size=500,  time_log=args.time_log)
        extractor.load_batches()
        extractor.query_batches_lit_specific(int(args.batch_start), int(args.batch_end))
        
        #python3 -m feature_extraction.literal_utils -e http://172.21.233/arql/ --dir /data/extracted_features_dbpedia2016 --lits_file literals.json --batch_start -1 --batch_end -1 distinct-literals
        #python3 -m feature_extraction.literal_utils -e httpi//172.21.233.14:8892/sparql/ --dir /data/extracted_features_dbpedia2016 --lits_file /data/extracted_features_dbpedia2016/literals_stat/batches_response_stats/literals.json --batch_start 1 --batch_end -1 extract-lits-stat

#python3 -m feature_extraction.literal_utils -e http://130.225.39.154:8892/sparql/ --dir /data/planrgcn_features/extracted_features_dbpedia2016 --lits_file /data/planrgcn_features/extracted_features_dbpedia2016/literals_stat/batches_response_stats/literals.json --batch_start 1 --batch_end -1 extract-lits-stat
