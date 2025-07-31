import pandas as pd
from nltk.tokenize import word_tokenize
from efficient_apriori import apriori

# Para imprimir muchas filas y columnas
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Bloque 1: Arules en Python

# Lectura de los datos en Excel
retail = pd.read_csv("Online Retail.txt", sep="\t", encoding="windows-1251")
print(retail.head(20))

# Armado de las baskets
retail = retail[['InvoiceNo', 'Description']].drop_duplicates().dropna()
retail = retail.groupby('InvoiceNo')['Description'].agg(tuple).reset_index()
print(retail.head(20))

retail_rules = apriori(retail["Description"].values.tolist(), min_support=0.005, min_confidence=0.5, verbosity=1)

# Se genera un dataframe para acceder
df_rules = []
for r in retail_rules[1]:
    df_rules.append({"lhs": r.lhs,
                     "rhs": r.rhs,
                     "support": r.support,
                     "confidence": r.confidence,
                     "lift": r.lift,
                     "lhs_len": len(r.lhs),
                     "rhs_len": len(r.rhs)})

df_rules = pd.DataFrame(df_rules)
print(df_rules)

print(df_rules.query('confidence > 0.8 and lhs_len > 1 and rhs_len == 1').nlargest(10, 'lift'))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Bloque 2: Manipulación de texto

# Read the Feather file
guia_oleo = pd.read_csv("guia_oleo.csv", sep="\t", encoding="utf-8")
guia_oleo = guia_oleo[guia_oleo['comment'].notna()]
print(guia_oleo.iloc[0])
print(guia_oleo.iloc[6])

# Tokenizo
guia_oleo['tokens'] = guia_oleo['comment'].apply(word_tokenize)    # Antes hay que ejecutar una única vez nltk.download('punkt')
print(guia_oleo['tokens'][0])

# Convert to lowercase each token
guia_oleo['tokens'] = guia_oleo['tokens'].apply(lambda x: [e.lower() for e in x])
print(guia_oleo['tokens'][0])

# Remuevo puntuacion
guia_oleo['tokens'] = guia_oleo['tokens'].apply(lambda x: [e for e in x if e.isalnum()])
print(guia_oleo['tokens'][0])

guia_oleo["comentario_malo"] = guia_oleo["clase_comentario"].map({"Malo": "clase_malo"}).fillna("clase_no_malo")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Arules de reviews

for i, row, in guia_oleo[['tokens', 'comentario_malo']].iterrows():
    row["tokens"].append(row["comentario_malo"])

# Encuentro las reglas de asociación
guia_oleo_rules = apriori(guia_oleo["tokens"].values.tolist(), min_support=0.01, min_confidence=0.2, max_length=3, verbosity=1)

# Generate a rules dataframe
df_guia_oleo_rules = []
for r in guia_oleo_rules[1]:
    df_guia_oleo_rules.append({"lhs": r.lhs,
                               "rhs": r.rhs,
                               "support": r.support,
                               "confidence": r.confidence,
                               "lift": r.lift,
                               "lhs_len": len(r.lhs),
                               "rhs_len": len(r.rhs)})

df_guia_oleo_rules = pd.DataFrame(df_guia_oleo_rules)
print(df_guia_oleo_rules)
print(df_guia_oleo_rules[(df_guia_oleo_rules["rhs_len"] == 1) & df_guia_oleo_rules["rhs"].apply(lambda x: "clase_no_malo" in x)].sort_values(by="lift", ascending=False).head(50))
print(df_guia_oleo_rules[(df_guia_oleo_rules["rhs_len"] == 1) & df_guia_oleo_rules["rhs"].apply(lambda x: "clase_malo" in x)].sort_values(by="lift", ascending=False).head(50))
