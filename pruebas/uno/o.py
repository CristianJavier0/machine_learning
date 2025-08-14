import pandas as pd
import csv

# 1. Lee tu CSV original (sin tildes, sin comillas manuales)
df = pd.read_csv("datos1.csv")

# 2. Exporta aplicando QUOTE_MINIMAL (añade comillas sólo donde hay comas o saltos)
df.to_csv(
    "datos2.csv",
    index=False,
    quoting=csv.QUOTE_MINIMAL
)