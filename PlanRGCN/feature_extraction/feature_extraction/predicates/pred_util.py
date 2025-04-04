import json
import os
import argparse
import time
from feature_extraction.sparql import Endpoint
import numpy as np


class ExtractorBase:
    def __init__(
        self, endpoint: Endpoint, output_dir: str, predicate_file="predicates.json"
    ) -> None:
        self.endpoint = endpoint
        self.output_dir = output_dir
        self.predicate_file = predicate_file

    def load_batches(self):
        
        if not hasattr(self, "batch_output_dir"):
            raise Exception(
                """Condition for this method to work is 
                \n\t1) the features (specifically batching of predicates of PredicateCoPredicateExtractor need to be run)
                \n\t2) the resulting 'batch_output_dir' needs to exist with the path to the batches"""
            )
        files = os.listdir(self.batch_output_dir)
        print(self.batch_output_dir)
        if len(files) == 0:
            self.batchify()
        sort_func = lambda x: int(x.split("ch_")[1].split(".json")[0])
        files = sorted([f"{self.batch_output_dir}/{x}" for x in files], key=sort_func)
        b = []
        for file in files:
            with open(file, "r") as f:
                b.append(json.load(f))
        self.batches = b
        # del b

    def batchify(self, batch_size=50):
        if not hasattr(self, "batch_output_dir"):
            raise Exception(
                """Condition for this method to work is spefifying the 'batch_output_dir' folder"""
            )
        if not hasattr(self, "predicates"):
            raise Exception("""Predicates must be loaded into the model""")
        if hasattr(self, "batch_size"):
            batch_size = self.batch_size
        if hasattr(self, "literals"):
            self.batchify_literals(batch_size)
            return
        
        print(len(self.predicates))
        preds = self.predicates
        batches = []
        while len(preds) > 0:
            t = preds[:batch_size]
            batches.append(t)
            with open(
                f"{self.batch_output_dir}/pred_in_batch_{len(batches)}.json", "w"
            ) as f:
                json.dump(t, f)
            preds = preds[batch_size:]
        assert np.sum([len(x) for x in batches]) == len(self.predicates)
        print(batches)
        self.batches = batches
    def batchify_literals(self, batch_size):
        print("Literals batch creation ", len(self.literals))
        lits = self.literals
        types = self.literal_types
        batches = []
        while len(lits) > 0:
            t = list(zip(lits[:batch_size], types[:batch_size]))
            batches.append(t)
            with open(
                f"{self.batch_output_dir}/pred_in_batch_{len(batches)}.json", "w"
            ) as f:
                json.dump(t, f)
            lits = lits[batch_size:]
            types = types[batch_size:]
        assert np.sum([len(x) for x in batches]) == len(self.literals)
        self.batches = batches


class PredicateExtractor(ExtractorBase):
    def __init__(
        self, endpoint: Endpoint, output_dir: str, predicate_file="predicates.json"
    ) -> None:
        super().__init__(endpoint, output_dir, predicate_file)

    def get_predicates_query(self):
        start = time.time()
        query = """SELECT DISTINCT ?p WHERE {
            ?s ?p ?o
        }
        """
        res = self.endpoint.run_query(query)
        end = time.time()
        duration = end - start
        with open(f"{self.output_dir}/predicate_result_time.txt", 'w') as f:
            f.write(f"Predicate extraction time: {duration}")
        res_fp = f"{self.output_dir}/predicate_result.json"
        json.dump(res, open(res_fp, "w"))
        print("Predicates extracted")
        predicates = []
        for bind in res["results"]["bindings"]:
            predicates.append(bind["p"]["value"])
        return predicates

    def save_predicates(self):
        preds = self.get_predicates_query()
        json.dump(preds, open(f"{self.output_dir}/{self.predicate_file}", "w"))
        print("Predicated Saved")


class PredicateCoPredicateExtractor(ExtractorBase):
    def __init__(
        self,
        endpoint: Endpoint,
        input_dir: str,
        output_dir: str,
        predicate_file="predicates.json",
    ) -> None:
        super().__init__(endpoint, output_dir, predicate_file)

        self.batch_output_dir = f"{output_dir}/batches"
        self.input_dir = input_dir
        self.predicates = json.load(
            open(f"{self.input_dir}/{self.predicate_file}", "r")
        )

    def query_batches(self, batch_start=1, batch_end=2):
        if not hasattr(self, "batches"):
            self.load_batches()
        save_path = f"{self.output_dir}/batch_response"
        f = open( f"{self.output_dir}/pred_co_time_log.txt" , 'a')
        os.system(f"mkdir -p {save_path}")
        print(f"Predicate Co-occurance stats are saved to: {save_path}")
        batch_end_idx = min(batch_end - 1, len(self.batches) - 1)
        if batch_end_idx == -1:
            batch_end_idx = len(self.batches) - 1
        for i, b in enumerate(self.batches[batch_start - 1 : batch_end_idx]):
            query = PredicateCoPredicateExtractor.query_gen(b)
            start = time.time()
            res = self.endpoint.run_query(query)
            dur = time.time()-start
            res_fp = f"{save_path}/batch_{batch_start+i}.json"
            json.dump(res, open(res_fp, "w"))
            print(f"batch {batch_start+i} extracted!")
            f.write(f"batch {batch_start+i}: {dur}\n")
            f.flush()
        f.close()

    def query_gen(batch):
        pred_str = ""
        for p in batch:
            if " " in p:
                raise Exception("space in pred")
            pred_str += f"(<{p}>) "
        query = f"""
        Select distinct ?p1, ?p2 where {{
            VALUES (?p1) {{ {pred_str}}}
            ?s ?p1 ?o .
            ?o ?p2 ?o2 . }}
        """
        return query


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Predicate Util",
        description="Utility for predicate feature extraction",
        epilog="Text at the bottom of help",
    )

    parser.add_argument("task")
    parser.add_argument("-e", "--endpoint")
    parser.add_argument("--dir", "--output_dir")
    parser.add_argument("--input_dir")
    parser.add_argument("--pred_file", default="predicates.json")
    parser.add_argument("--batch_start")
    parser.add_argument("--batch_end")

    args = parser.parse_args()

    if args.task == "extract-predicates":
        output_dir = f"{args.dir}/predicate"
        os.system(f"mkdir -p {output_dir}")
        endpoint = Endpoint(args.endpoint)
        pred_ex = PredicateExtractor(
            endpoint, output_dir, predicate_file=args.pred_file
        )
        pred_ex.save_predicates()
    elif args.task == "extract-co-predicates":
        endpoint = Endpoint(args.endpoint)
        input_dir = f"{args.input_dir}/predicate"
        os.system(f"mkdir -p {args.dir}")
        extrator = PredicateCoPredicateExtractor(
            endpoint, input_dir, args.dir, args.pred_file
        )
        # extrator.batchify()
        extrator.query_batches(int(args.batch_start), int(args.batch_end))
