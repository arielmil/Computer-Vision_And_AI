import torch
import torch.nn as nn
import torch.nn.functional as F
from neat.nn import FeedForwardNetwork

class Genome:
    def __init__(self, genome, config):
        """
        Armazena o genome do NEAT (objeto neat-python)
        e a configuração do NEAT, para uso posterior.
        """
        self.genome = genome
        self.config = config
    
    def get_topological_info(self):
        """
        Cria um FeedForwardNetwork usando neat-python, 
        só para extrair as informações de topologia:
          - A lista completa em ordem topológica (node_order)
          - Quais são os IDs dos nós de entrada (input_nodes)
          - Quais são os IDs dos nós de saída (output_nodes)

        Retornamos (node_order, input_nodes, output_nodes).
        """
        # Cria a rede feedforward (padrão da lib NEAT) com base neste genome e config
        net = FeedForwardNetwork.create(self.genome, self.config)
        
        # O net.node_evals é processado na ordem topológica, mas primeiramente
        # guardamos os nós de entrada numa lista:
        node_order = list(net.input_nodes)

        # Depois percorremos os node_evals para continuar a ordem topológica
        for node_eval in net.node_evals:
            node_id = node_eval[0]
            node_order.append(node_id)
        
        # Por fim, verifica se todos os nós de saída estão na lista node_order
        for output in net.output_nodes:
            if output not in node_order:
                node_order.append(output)
        
        # Por fim, retornamos a lista completa (node_order),
        # junto com os input_nodes e output_nodes originais:
        return node_order, net.input_nodes, net.output_nodes

    def decode_genome_to_torch(self):
        """
        Constrói dinamicamente um nn.Module PyTorch
        que respeita a topologia definida pelo NEAT.
        """

        # Extrai as informações de topologia:
        #   node_order  = lista completa em ordem topológica
        #   in_nodes    = IDs de nós de entrada
        #   out_nodes   = IDs de nós de saída
        node_order, in_nodes, out_nodes = self.get_topological_info()

        # Após extrair as informações de topologia, garante que todos os nós nas conexões estão presentes na lista node_order
        for cg in self.genome.connections.values():
            if cg.enabled:
                in_node, out_node = cg.key
                if in_node not in node_order:
                    node_order.append(in_node)
                if out_node not in node_order:
                    node_order.append(out_node)

        # Vamos referenciá-las localmente para passar ao construtor da classe interna:
        genome = self.genome
        config = self.config
        topo_order = node_order

        class NeatModule(nn.Module):
            def __init__(_self, genome, config, node_order, in_nodes, out_nodes):
                """
                _self => é a instância do NeatModule em si.
                """
                super().__init__()
                
                # Armazena objetos e listas
                _self.genome = genome
                _self.config = config
                _self.node_order = node_order
                _self.input_nodes = in_nodes
                _self.output_nodes = out_nodes

                # Coletamos as conexões habilitadas (arestas) do genome:
                _self.connections = []
                for cg in _self.genome.connections.values():
                    if cg.enabled:
                        in_node, out_node = cg.key
                        w = cg.weight
                        # Salva tuplas (nó de entrada, nó de saída, peso)
                        _self.connections.append((in_node, out_node, w))

                # Esses valores vêm do config NEAT (ex.: 400 inputs, 10 outputs):
                _self.num_inputs = _self.config.genome_config.num_inputs
                _self.num_outputs = _self.config.genome_config.num_outputs
                
                # Mapeia (node_id) -> índice no vetor de ativações
                _self.node_id_to_idx = {}
                for i, n_id in enumerate(_self.node_order):
                    _self.node_id_to_idx[n_id] = i

            def forward(_self, x):
                """
                Executa o forward pass da topologia NEAT:
                  1. Alimenta os nós de entrada com as colunas de x
                  2. Propaga para nós ocultos e nós de saída na ordem topológica
                  3. Retorna um tensor [batch_size, num_outputs]
                """
                # x tem shape [batch_size, num_features],
                # onde num_features deve bater com num_inputs configurado no NEAT.

                batch_size = x.shape[0]
                total_nodes = len(_self.node_order)

                # Cria um tensor de ativações zerado para todos os nós
                activations = x.new_zeros((batch_size, total_nodes))

                # (1) Alimentar as ativações dos nós de entrada
                # Em vez de assumir que vão de 0..num_inputs-1,
                # pegamos o array de IDs de entrada (input_nodes) do NEAT.
                for col_index, node_id in enumerate(_self.input_nodes):
                    # Acha o índice no vetor de ativações correspondente a esse node_id
                    idx = _self.node_id_to_idx[node_id]
                    # Copia a coluna col_index de x para esse índice
                    activations[:, idx] = x[:, col_index]

                # (2) Construir dicionário adjacency: para cada nó destino,
                #     lista de (nó de origem, peso).
                adjacency = {n_id: [] for n_id in _self.node_order}
                for (i_node, o_node, w) in _self.connections:
                    if i_node not in _self.node_id_to_idx or o_node not in _self.node_id_to_idx:
                        continue
                    adjacency[o_node].append((i_node, w))

                # (3) Processar os nós (ocultos + saída) na ordem topológica
                #     ignorando os nós de entrada (já preenchidos).
                #     Lembrando que node_order contém: 
                #       [input_nodes ... hidden_nodes ... output_nodes]
                #     Portanto, começamos de _self.num_inputs até o fim:
                for node_id in _self.node_order[_self.num_inputs:]:
                    node_idx = _self.node_id_to_idx[node_id]
                    # Soma ponderada das ativações dos nós de origem
                    total_in = torch.zeros(batch_size, device=x.device)
                    for (src_id, w) in adjacency[node_id]:
                        src_idx = _self.node_id_to_idx[src_id]
                        total_in += activations[:, src_idx] * w

                    # Exemplo de ativação interna: Softmax por nó (não usual, mas demonstrativo).
                    # Se preferir ReLU: node_out = F.relu(total_in)
                    node_out = F.softmax(total_in, dim=0)
                    activations[:, node_idx] = node_out

                # (4) Extrair as saídas. Em vez de assumir IDs 400..409 ou algo do gênero,
                #     usamos os node_ids de output_nodes que a própria rede NEAT criou.
                outs = []
                for node_id in _self.output_nodes:
                    out_idx = _self.node_id_to_idx[node_id]
                    outs.append(activations[:, out_idx])

                # Retorna as saídas empilhadas => shape [batch_size, num_outputs]
                return torch.stack(outs, dim=1)

        # Retornamos uma instância do NeatModule, passando as listas obtidas
        return NeatModule(self.genome, self.config, topo_order, in_nodes, out_nodes)


def eval_genomes(genomes, config, X_train, y_train):
    """
    Função de avaliação que o NEAT chama em cada geração.
    Recebe:
      - genomes: lista de (genome_id, neat_genome_obj)
      - config: objeto de configuração do NEAT
      - X_train, y_train: dados de treino (Numpy arrays)

    Cria um módulo PyTorch a partir de cada genome e mede a performance (fitness).
    """

    # Converte X e y para tensores no device (GPU se disponível)
    X_train_torch = torch.tensor(X_train, dtype=torch.float, device='cuda')
    y_train_torch = torch.tensor(y_train, dtype=torch.long, device='cuda')

    for genome_id, genome in genomes:
        # Cria nossa classe "Genome" que sabe montar uma rede PyTorch
        _genome = Genome(genome, config)

        # Constrói a rede PyTorch (nn.Module)
        net = _genome.decode_genome_to_torch().to("cuda")

        # Forward pass em todo o dataset (cuidado com memória se for grande)
        outputs = net(X_train_torch)  # shape [batch_size, num_outputs]

        # Exemplo de perda: cross-entropy (usando log_softmax manual)
        # Se preferir, pode usar F.cross_entropy(outputs, y_train_torch) diretamente
        log_probs = F.log_softmax(outputs, dim=1)
        loss = -log_probs[range(len(y_train_torch)), y_train_torch].mean()

        # O NEAT maximiza o fitness, então definimos fitness = -loss
        # (quanto menor a perda, maior o fitness)
        genome.fitness = float(-loss.detach().cpu().numpy())
