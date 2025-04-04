from feature_extraction.predicates.pred_util import *
import sys


class EntityExtractor(ExtractorBase):
    def __init__(
        self,
        endpoint: Endpoint,
        output_dir: str,
        entity_file="entity.json",
    ) -> None:
        super().__init__(endpoint, output_dir, entity_file)
        self.entity_file = entity_file
        if self.output_dir.endswith("/"):
            self.output_dir = self.output_dir[:-1]
        self.query()

    def get_query():
        return """ 
             SELECT DISTINCT ?entity
             WHERE {
            { ?entity ?predicate ?object }
            UNION
            { ?subject ?predicate ?entity }
            } 
        """

    def query(self):
        save_path = self.output_dir
        os.system(f"mkdir -p {save_path}")
        print(f"Entities are saved to: {save_path}")
        query = EntityExtractor.get_query()
        start = time.time()
        res = self.endpoint.run_query(query)
        dur = time.time() - start
        res_fp = f"{save_path}/entities_response.json"
        json.dump(res, open(res_fp, "w"))
        res = res["results"]["bindings"]
        res = [x["entity"]["value"] for x in res]
        res_fp = f"{save_path}/entities.json"
        json.dump(res, open(res_fp, "w"))
        with open(f"{self.output_dir}/entity_result_time.txt", 'w') as f:
            f.write(f"Predicate extraction time: {dur}")
        print(f"entities extracted!")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        help_text = """
        Provided arguments are insufficient!
        - SPARQL endpoint url
        - Output dir path where the entities in KG should be extracted to
        """
        print(help_text)
        exit()
    endpoint_url = sys.argv[1]
    endpoint = Endpoint(endpoint_url)
    output_dir = sys.argv[2]
    text = f"""Endpoint url : {endpoint_url}
    Output dir: {output_dir}
    """
    print(text)
    EntityExtractor(endpoint=endpoint, output_dir=output_dir)
