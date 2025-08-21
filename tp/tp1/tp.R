# Instalar tinytex si no está instalado
if (!require(tinytex)) {
  install.packages("tinytex")
  tinytex::install_tinytex()
}
# Instalar paquetes si no están instalados
if (!require(rpart)) install.packages("rpart", dependencies = TRUE)
if (!require(rpart.plot)) install.packages("rpart.plot", dependencies = TRUE)
if (!require(caret)) {
  message("Intentando instalar 'caret'...")
  install.packages("caret", dependencies = TRUE)
  if (!require(caret)) stop("No se pudo instalar 'caret'. Prueba 'install.packages(\"caret\", repos = \"https://cloud.r-project.org\", type = \"source\", dependencies = TRUE)' en la consola.")
}

# Cargar las librerías necesarias
library(rpart)
library(rpart.plot)

# --- PASO 1: Cargar y preparar los datos ---
# Leemos el archivo directamente desde una URL
adult_df <- read.csv("adult.csv", 
                     header = FALSE, 
                     col.names = c("age", "workclass", "fnlwgt", "education", "education.num", "marital.status", "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss", "hours.per.week", "native.country", "income"), 
                     na.strings = "?")

# Eliminamos las filas con valores faltantes
adult_df <- na.omit(adult_df)

# PASO 2: Crear el árbol de decisión
arbol_basico <- rpart(income ~ ., data = adult_df, method = "class")

# PASO 3: Visualizar el árbol de decisión
rpart.plot(arbol_basico, extra = 101, main = "Mi Primer Árbol de Decisión")