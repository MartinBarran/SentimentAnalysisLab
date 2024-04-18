import pandas as pd
from pysentimiento import create_analyzer

analyzer = create_analyzer(task="sentiment", lang="es")


input_file = 'input_archivoTipo.csv' # Cambiar nombre de archivo input de ser necesario
output_file = 'output.csv'
df = pd.read_csv(input_file)


sentiments = []
probas_list = []


for text in df['content']: # Se espera una columna con encabezado "content"
    output = analyzer.predict(text)
    sentiments.append(output.output)
    probas_list.append(output.probas)


df['sentiment'] = sentiments
df['probas'] = probas_list


df.to_csv(output_file, index=False)
