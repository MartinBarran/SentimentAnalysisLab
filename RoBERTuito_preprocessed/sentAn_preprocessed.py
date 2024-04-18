import pandas as pd
from pysentimiento import create_analyzer
from pysentimiento.preprocessing import preprocess_tweet

analyzer = create_analyzer(task="sentiment", lang="es")

input_file = 'input_archivoTipo.csv'
output_file = 'output.csv'
df = pd.read_csv(input_file)

sentiments = []
probas_list = []


for text in df['content']:
    if isinstance(text, float):
        text = str(text)
        
    output = analyzer.predict(preprocess_tweet(text))
    sentiments.append(output.output)
    probas_list.append(output.probas)


df['sentiment'] = sentiments
df['probas'] = probas_list

df.to_csv(output_file, index=False)
#print(preprocess_tweet("@rodanmars @ContciudaDBJ @mileniotv @CiroGomezL @STaboadaMx @AAtaydeR @betotok @marianagc @GabyCuevas @vicmeac @LuisMendozaBJ @BJAlcaldia @FiscaliaCDMX @ra_taboada_mx @RicardoBSalinas @lopezobrador_ @AccionNacional Tu síguete exhibiendo! Para que todos los vecinos hamburguesados (sic.) de la Del Valle se den cuenta y se convenzan que ni de p3d0 morena sirve #NarcoPresidenteAMLO15"))