import protobuf  # 确保你已经安装了protobuf库  
import pyarrow.parquet as pq  
import geopandas as gpd  
import psycopg2  
from sqlalchemy import create_engine  
import sqlite3  
import pandas as pd  
from sqlalchemy import text 
  
# 指定Parquet文件的路径  
# file_path = './part-00031-b09b2e77-9810-42da-88a4-45d93a140a19-c000.snappy.parquet'  
file_path = './part-00031-79e16b9a-a970-4e53-b534-59380a3a5e58-c000.snappy.parquet'  

  
# 打开Parquet文件  
parquet_file = pq.ParquetFile(file_path)  
  
# 获取文件的元数据信息  
print("File Metadata:")  
print(parquet_file.metadata)  
  
# 读取文件的第一个数据分区（如果有多个分区的话）  
# 注意：根据你的数据结构和需求，你可能需要修改这部分代码来正确处理你的数据  
data = parquet_file.read_row_group(0)  
  
# 将数据转换为Pandas DataFrame（可选）  
df = data.to_pandas()  
  
# 显示数据的前几行（可选）  
print("Data Preview:")  
print(df.head(5))

# print(df.head(50))
# print(df['laneBoundary'])

# 将DataFrame写入JSON文件  
json_file_path = 'read_mdc.json'  # 指定JSON文件的路径和名称  
# df.to_json(json_file_path, orient='records', indent=4)  # 使用orient参数指定JSON格式，indent参数指定缩进级别
# df = pd.read_json(json_file_path) 
  
# 创建数据库引擎  
# 这里以SQLite为例，你也可以使用其他数据库，如MySQL、PostgreSQL等  
# 只需更改数据库连接字符串即可  
engine = create_engine('sqlite:///mdc_db.db')  
  
# 如果你的数据在DataFrame中，使用以下代码将数据写入数据库  
# df.to_sql('table_name', engine, if_exists='replace', index=False)  
  
# 如果你的数据在JSON文件中，首先读取JSON文件并将其转换为DataFrame  
# 然后使用相同的to_sql()方法将数据写入数据库  
 
df.to_sql('mdc_table', engine, if_exists='replace', index=False)
  
# 连接到数据库  
conn = engine.connect()
  
# 创建一个游标对象  
from sqlalchemy.orm import sessionmaker  
  
Session = sessionmaker(bind=engine)  
session = Session()
# cursor = conn.cursor()  
  

    
sqltext = f"PRAGMA table_info(mdc_table)"
result = session.execute(text(sqltext))  

for row in result:  
    print(row)
    
# 执行查询语句  
sqltext = "SELECT distinct(carmodel) FROM mdc_table"
 
result = session.execute(text(sqltext))  
for row in result:  
    print(row)
    
# 执行查询语句  
sqltext = "SELECT distinct(dataformat) FROM mdc_table"
 
result = session.execute(text(sqltext))  
for row in result:  
    print(row)
    
session.close()  
engine.dispose()

# 替换 'SELECT * FROM table_name' 为你自己的查询语句  
# query = 'SELECT * FROM mdc_table'  
# cursor.execute(query)  

# # 获取查询结果  
# results = cursor.fetchall()  
  
# # 处理查询结果  
# for row in results:  
#     # 打印每行数据  
#     print(row)  
  
# # 关闭游标和连接  
# cursor.close()  
# conn.close()

# 将DataFrame或JSON转换为GeoDataFrame  
# 如果已经是GeoDataFrame，则可以跳过此步骤  
# 这里假设你的DataFrame保存在变量df中，且包含地理空间列'geometry'  
# gdf = gpd.GeoDataFrame(df, geometry='geometry')  

# # 设置数据库连接参数  
# db_name = 'mdc_geopandas'  
# db_user = 'wqc'  
# db_password = '123'  
# db_host = 'localhost'  
# db_port = '5432'  
  
# # 创建数据库连接字符串  
# conn_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"  
  
# # 创建数据库引擎  
# engine = create_engine(conn_string)  
  
# # 将GeoDataFrame写入PostGIS空间数据库  
# gdf.to_postgis('table_name', engine, if_exists='replace', index=False)

def is_protobuf_format(file_path):  
    try:  
        with open(file_path, 'rb') as f:  
            protobuf.parse(f.read())  # 尝试解析数据  
        return True  
    except Exception as e:  
        return False  
# print(r'./part-00031-b09b2e77-9810-42da-88a4-45d93a140a19-c000.snappy.parquet')
# file_path = r"./part-00031-b09b2e77-9810-42da-88a4-45d93a140a19-c000.snappy.parquet"  # 替换为你的数据文件路径  
# file_path = r"./protobuf/sensoris/protobuf/categories/weather.proto"  # 替换为你的数据文件路


# if is_protobuf_format(file_path):  
#     print("数据是protobuf格式")  
# else:  
#     print("数据不是protobuf格式")