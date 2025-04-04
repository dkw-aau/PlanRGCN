from datetime import datetime
import sys
from feature_extraction.predicates.pred_util import ExtractorBase
from feature_extraction.sparql import Endpoint
import os
import json
from urllib.parse import urlparse

import pytz


class PredEntExtractor(ExtractorBase):
    def __init__(
        self,
        endpoint: Endpoint,
        output_dir: str,
        predicate_file: str,
    ) -> None:
        super().__init__(endpoint, output_dir, None)

        self.output = f"{output_dir}/pred_ent/batch_response"
        os.system(f"mkdir -p {self.output}")
        self.predicates = json.load(open(predicate_file, "r"))

    def query(self, quantile=0.9):
        save_path = self.output
        extracted_preds = []
        for f in os.listdir(save_path):
            pred = json.load(open(f"{save_path}/{f}"))["predicate"]
            extracted_preds.append(pred)
        denmark_timezone = pytz.timezone("Europe/Copenhagen")
        print(f"Entity Predicate Stats are saved to: {save_path}")

        for i, p in enumerate(self.predicates):
            if p in extracted_preds:
                continue
            current_datetime_denmark = datetime.now(denmark_timezone)
            # Format and print the datetime
            formatted_datetime = current_datetime_denmark.strftime(
                "%Y-%m-%d %H:%M:%S %Z"
            )
            print(f"{i}: Starting on {p} ({formatted_datetime})")
            query = PredEntQueries.get_top_ent(p, quantile=quantile)
            res = self.endpoint.run_query(query)
            # if i == 1:
            #    print(res)
            res["predicate"] = p
            filename = os.path.basename(urlparse(p).path)
            res_fp = f"{save_path}/{filename}.json"
            json.dump(res, open(res_fp, "w"))


class PredEntQueries:
    def get_top_ent(predicate_iri, quantile=0.9):
        # SPARQL query template
        sparql_query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

        SELECT ?entity (COUNT(?entity) as ?count)
        WHERE {{
        ?entity <{predicate_iri}> ?object.
        }}
        GROUP BY ?entity
        HAVING (COUNT(?entity) >= {quantile} * (SELECT (COUNT(?entity) as ?maxCount)
                                    WHERE {{
                                    ?entity <{predicate_iri}> ?object.
                                    }}
                                    GROUP BY ?predicate))
        ORDER BY DESC(?count)
        """

        return sparql_query


if __name__ == "__main__":
    if len(sys.argv) < 5:
        help_text = """
        Missing arguments
        Please specify:
        endpoint: Endpoint,
        output_dir: str,
        predicate_file="predicates.json",
        """
        print(help_text)
        print(sys.argv[1:])
    endpoint = Endpoint(sys.argv[1])
    output_dir = sys.argv[2]
    pred_file = sys.argv[3]
    quantile = float(sys.argv[4])
    print(
        f"""
          endpoint      : {sys.argv[1]}
          dir           : {output_dir}
          pred_file     : {pred_file}
          quantile      : {quantile}
          """
    )
    p = PredEntExtractor(endpoint, output_dir, pred_file)
    p.query(quantile=quantile)
