import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus

# Read CSV
df = pd.read_csv("retail_price.csv")

# Clean column names
df.columns = df.columns.str.lower().str.replace(" ", "_")

# MySQL credentials
username = "root"
password = quote_plus("Varun@123")   # Handles @ properly
host = "localhost"
port = 3306
database = "retail_db"

# Create connection
engine = create_engine(
    f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
)

# Upload CSV to MySQL
df.to_sql(
    name="sales_data",
    con=engine,
    if_exists="replace",
    index=False
)

print("✅ CSV imported successfully into MySQL!")