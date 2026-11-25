import sqlite3, csv, sys
from contextlib import closing
class Inventory:
    def __init__(self,path):
        self.path=path
        with closing(sqlite3.connect(path)) as c, c: c.execute('CREATE TABLE IF NOT EXISTS products(name TEXT PRIMARY KEY, stock INTEGER NOT NULL CHECK(stock>=0))')
    def register(self,name,stock=0):
        if not isinstance(name,str) or not 0<len(name.strip())<=100 or type(stock)!=int or stock<0: raise ValueError('Invalid product')
        with closing(sqlite3.connect(self.path)) as c,c: c.execute('INSERT INTO products VALUES (?,?)',(name,stock))
    def adjust(self,name,delta):
        if type(delta)!=int: raise ValueError('Integer required')
        with closing(sqlite3.connect(self.path)) as c,c:
            if c.execute('UPDATE products SET stock=stock+? WHERE name=?',(delta,name)).rowcount!=1: raise ValueError('Unknown product')
    def search(self,term=''):
        with closing(sqlite3.connect(self.path)) as c: return c.execute('SELECT name,stock FROM products WHERE instr(lower(name),lower(?))>0 ORDER BY name',(term,)).fetchall()
    def export(self,path):
        with open(path,'w',newline='',encoding='utf-8') as f:
            w=csv.writer(f); w.writerow(['name','stock']); w.writerows(self.search())
if __name__=='__main__':
    i=Inventory('inventory.sqlite3')
    while True:
        try:
            action=input('register / adjust / search / export / quit: ')
            if action=='quit': break
            if action=='register': i.register(input('Name: '),int(input('Stock: ')))
            elif action=='adjust': i.adjust(input('Name: '),int(input('Change: ')))
            elif action=='search': print(i.search(input('Search: ')))
            elif action=='export': i.export('inventory.csv')
        except (EOFError,KeyboardInterrupt): break
        except Exception as e: print('Error:',e)
