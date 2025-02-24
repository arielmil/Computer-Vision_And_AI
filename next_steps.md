## Passos que faltam: ##

# 1: 

    - Reduzir a qualidade da foto para ser uma X = 20 por Y = 20 para ter menos nós de entrada. (20 x 20): OK

# 2:

    - Atualizar o pipeline para receber fotos com menos nós de entrada: OK

# 3:

    - Fazer uma separação de modulos: OK

        - Um modulo cuida de baixar as fotos e carregar para outros modulos através da variavel path (com o path do dataset): OK
        - Outro ficara responsável em fazer essa redimensionalização das fotos do dataset: OK
        - Outro tera a implementação do algorítimo que roda o NEAT em si: OK
        - E o ultimo será a main que juntará tudo e colocará para funcionar: OK

# 4:

    - Fazer arquivos de testes para todos esses modulos (menos para a main).
    - Recomeçar o projeto do 0, começando dos arquivos de testes.

# 5:

    - Fazer um package com esse projeto dentro da veenv tensorneat, para facilitar o gerenciamento de imports do projeto.
    - Fazer um encapsulamento do projeto com um __init__.py.
    