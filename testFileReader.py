import configparser
import os

# Caminho do arquivo de configuração no diretorio atual:

#muda para o diretorio Computer vision
os.chdir("Computer vision")

ini_path = r"config.ini"
config_test = configparser.ConfigParser()
config_test.read(ini_path, encoding="utf-8")  # ou 'utf-8-sig' se suspeitar BOM

print("Seções encontradas:", config_test.sections())

# Tente imprimir itens da seção [DefaultSpeciesSet]
if "DefaultSpeciesSet" in config_test:
    print("Itens em [DefaultSpeciesSet]:")
    for k,v in config_test["DefaultSpeciesSet"].items():
        print(f"  {k} = {v}")
else:
    print("[DefaultSpeciesSet] não foi encontrado!")
