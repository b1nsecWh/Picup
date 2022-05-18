import sqlite3

class binaryApi2soDB():
    ''' the data base save the dependency information form API to Function'''

    def __init__(self,db_path,binary_name):
        self.table_name=binary_name
        self.conn=sqlite3.connect(db_path)
    
    def close(self):
        self.conn.close()
    
    def search_by_name(self,name):
        try:
            c=self.conn.cursor()
            cursor=c.execute("SELECT * FROM {} Where name='{}'".format(self.table_name,name))
            
            rst=list(cursor)

            return rst
        except sqlite3.Error as e:
            print(e)