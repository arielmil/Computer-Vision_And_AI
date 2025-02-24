## Passos que faltam: ##

# 1: 

    - Reduzir a qualidade da foto para ser uma X por Y para ter menos nós de entrada.

# 2:

    - Atualizar o pipeline para receber fotos com menos nós de entrada.

# 3:

    - Fazer uma separação de modulos:

        - Um modulo cuida de baixar as fotos e carregar para outros modulos através da variavel path (com o path do dataset).
        - Outro ficara responsável em fazer essa redimensionalização das fotos do dataset.
        - Outro tera a implementação do algorítimo que roda o NEAT em si.
        - E o ultimo será a main que juntará tudo e colocará para funcionar.

# 4:

    - Fazer arquivos de testes para todos esses modulos (menos para a main).
    - Recomeçar o projeto do 0, começando dos arquivos de testes.

# 5:

    - Fazer um package com esse projeto dentro da veenv tensorneat, para facilitar o gerenciamento de imports do projeto.
    - Fazer um encapsulamento do projeto com um __init__.py.
    