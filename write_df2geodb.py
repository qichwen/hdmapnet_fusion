import geopandas as gpd  
import psycopg2  
from sqlalchemy import create_engine  

# 将DataFrame或JSON转换为GeoDataFrame  
# 如果已经是GeoDataFrame，则可以跳过此步骤  
# 这里假设你的DataFrame保存在变量df中，且包含地理空间列'geometry'  
gdf = gpd.GeoDataFrame(df, geometry='geometry')  
  
# 设置数据库连接参数  
db_name = 'your_database_name'  
db_user = 'your_username'  
db_password = 'your_password'  
db_host = 'your_host'  
db_port = 'your_port'  
  
# 创建数据库连接字符串  
conn_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"  
  
# 创建数据库引擎  
engine = create_engine(conn_string)  
  
# 将GeoDataFrame写入PostGIS空间数据库  
gdf.to_postgis('table_name', engine, if_exists='replace', index=False)