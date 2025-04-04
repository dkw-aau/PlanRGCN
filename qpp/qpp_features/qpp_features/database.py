import sqlite3
import numpy as np

class DatabaseConnector:
    def __init__(self, file_name = "ged_wikidata.db") -> None:
        self.file_name = file_name
        self.con = sqlite3.connect(self.file_name)
        self.cursor = self.con.cursor()
        try:
            self.create_table()
        except sqlite3.OperationalError:
            print("ged table already exist and is therefore not created again.")
        
          
    def checkpoint(self):
        self.con.commit()
    
    def create_table(self):
        query = "CREATE TABLE ged(queryid1 varchar(255),queryid2 varchar(255),dist float, PRIMARY KEY (queryid1,queryid2));"
        self.cursor.execute(query)
        createSecondaryIndex = "CREATE INDEX id1 ON ged(queryid1, queryid2)"
        self.cursor.execute(createSecondaryIndex)
        self.con.commit()
    
        
    def insert_ids(self,queryid1, queryid2):
        try:
            self.cursor.execute(f"INSERT INTO ged VALUES ('{queryid1}', '{queryid2}', null)")
        except sqlite3.IntegrityError:
            return False
        try:
            self.cursor.execute(f"INSERT INTO ged VALUES ('{queryid2}', '{queryid1}', null)")
        except sqlite3.IntegrityError:
            return False
        self.con.commit()
        return True

    def insert_ged(self,queryid1, queryid2, dist):
        try:
            self.cursor.execute(f"INSERT INTO ged VALUES ('{queryid1}', '{queryid2}', {dist})")
        except sqlite3.IntegrityError:
            return False
        try:
            self.cursor.execute(f"INSERT INTO ged VALUES ('{queryid2}', '{queryid1}', {dist})")
        except sqlite3.IntegrityError:
            return False
        self.con.commit()
        return True
    
    def exists(self, queryid1, queryid2):
        res = self.cursor.execute(f"SELECT * FROM ged WHERE queryid1 == '{queryid1}' AND queryid2 == '{queryid2}'")
        if res.fetchone() is None:
            return False
        return True
    
    def retrieve_all(self):
        query="SELECT queryid1, queryid2, dist FROM ged"
        res = self.cursor.execute(query)
        return res.fetchall()
    
    def insert_multiple(self, data):
        """_summary_

        Args:
            data (_type_): [('queryid1', 'queryid2', dist), next value]
        """
        self.cursor.execute(f"INSERT INTO ged VALUES (?, ?, ?)", data)
        self.con.commit()
        pass
    def close(self):
        self.con.commit()
        self.con.close()
    def __del__(self):
        self.close()
    
    def create_query_table(self):
        query = "CREATE TABLE query(queryid varchar(255),queryString varchar(600), PRIMARY KEY (queryid));"
        self.cursor.execute(query)
        createSecondaryIndex = "CREATE INDEX id2 ON ged(queryid)"
        self.cursor.execute(createSecondaryIndex)
        
    def test_query_table(self):
        query_id = 5
        queryStr = """
        SELECT * WHERE {
            ?s ?p ?o .
        }
        """
        query = f"INSERT INTO query VALUES ('{query_id}', '{queryStr}')"
        self.cursor.execute(query)
        
class GEDDatabase:
    """ Goal of this class is to facilitate a faster lookup for
    """
    def __init__(self, file_name = None, update_improve=True, create_table = False) -> None:
        if file_name is None:
            raise Exception("You must specify a valide database file path")
            
        
        self.file_name = file_name
        self.update_improve = update_improve
        if update_improve:
            self.con = sqlite3.connect(self.file_name, isolation_level='DEFERRED')
            self.con.cursor().execute('''PRAGMA synchronous = EXTRA;''')
            self.con.cursor().execute('''PRAGMA journal_mode = OFF;''')
            self.con.cursor().execute('''PRAGMA cache_size = -4882813;''')
        else:
            self.con = sqlite3.connect(self.file_name)
        self.cursor = self.con.cursor()
        if create_table:
            try:
                self.create_table()
            except sqlite3.OperationalError:
                print("ged table already exist and is therefore not created again.")
        
          
    def checkpoint(self):
        self.con.commit()
    
    def create_table(self):
        query = "CREATE TABLE ged(query_pair varchar(510), queryid1 varchar(255),queryid2 varchar(255),dist real, PRIMARY KEY (query_pair));"
        
        
        self.cursor.execute(query)
        self.con.commit()

    def insert_ged(self,queryid1, queryid2, dist):
        try:
            self.cursor.execute(f"INSERT INTO ged VALUES ('{queryid1}_{queryid2}' ,'{queryid1}', '{queryid2}', {dist})")
        except sqlite3.IntegrityError:
            return False
        #self.con.commit() #manually commit/checkpoint because this can cause performance issues.
        return True
    
    def exists(self, queryid1, queryid2):
        res = self.cursor.execute(f"SELECT * FROM ged WHERE query_pair == \"{queryid1}_{queryid2}\"")
        if res.fetchone() is None:
            res1 = self.cursor.execute(f"SELECT * FROM ged WHERE query_pair == \"{queryid2}_{queryid1}\"")
            if res1.fetchone() is None:
                return False
        return True
    
    def get(self, queryid1, queryid2):
        res = self.cursor.execute(f"SELECT * FROM ged WHERE query_pair == \"{queryid1}_{queryid2}\"")
        fetched = res.fetchone()
        if fetched is None:
            res2 = self.cursor.execute(f"SELECT * FROM ged WHERE query_pair == \"{queryid2}_{queryid1}\"")
            fetched2 = res2.fetchone()
            if fetched2 is None:
                return np.iinfo(np.int16).max
            if fetched2[-1] == -1:
                return np.iinfo(np.int16).max
            else:
                return fetched2[-1]
        if fetched[-1] == -1:
            return np.iinfo(np.int16).max
        else:
            return fetched[-1]
    
    def retrieve_all(self):
        query="SELECT queryid1, queryid2, dist FROM ged"
        res = self.cursor.execute(query)
        return res.fetchall()
    
    def insert_multiple(self, data):
        """_summary_

        Args:
            data (_type_): [('queryid1', 'queryid2', dist), next value]
        """
        self.cursor.execute(f"INSERT INTO ged VALUES (?, ?, ?)", data)
        self.con.commit()
        pass
    def close(self):
        self.con.commit()
        self.con.close()
    
    def __del__(self):
        self.close()

class GEDDict(dict):
    def __init__(self, file_name = None, update_improve=True):
        self.db = GEDDatabase(file_name=file_name, update_improve = update_improve,create_table=False)
    def __setitem__(self,idx, item):
        raise Exception("Set operation should not happen")
        
    def __getitem__(self,idx):
        if not isinstance(idx, tuple):
            raise Exception(f"{idx} is not a tuple!")
        if len(idx) != 2:
            raise(f"Index should contain a pair of ids for queries and not {idx}")
        return self.db.get(idx[0],idx[1])
    
    def keys(self):
        raise Exception("keys operation should not happen")
        return []
    def values(self):
        raise Exception("values operation should not happen")
        return []
    def items(self):
        raise Exception("items operation should not happen")
        return []
    def get(self, idx):
        return self.__getitem__(idx)
   
if __name__ == "__main__":
    import sys, json,os
    dist_fold = sys.argv[1]
    db_file = sys.argv[2]
    db = GEDDatabase(file_name=db_file, create_table=True)
    files = os.listdir(dist_fold) 
    for n,i in enumerate(files):
        if not i.endswith("json"):
            continue
        print(f"beginning {n} of {len(files)}: {dist_fold}/{i}")
        dat = json.load(open(os.path.join(dist_fold,i),'r'))
        for y in dat:
            print(y)
            if y['dist'] == 'None':
                db.insert_ged(y['queryID1'], y['queryID2'], 1000000000)
            else:
                db.insert_ged(y['queryID1'], y['queryID2'], y['dist'])
        db.checkpoint()
