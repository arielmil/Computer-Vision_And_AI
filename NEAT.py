import torch
import torch.nn as nn
import torch.nn.functional as F
from neat.nn import FeedForwardNetwork
import math

iterations = 0

class Genome:
    def __init__(self, genome, config):
        """
        Armazena o genome do NEAT (objeto neat-python)
        e a configuração do NEAT, para uso posterior.
        """
        self.genome = genome
        self.config = config
    
    def get_topological_info(self, debug = False):
        """
        Cria um FeedForwardNetwork usando neat-python, 
        só para extrair as informações de topologia:
          - A lista completa em ordem topológica (node_order)
          - Quais são os IDs dos nós de entrada (input_nodes)
          - Quais são os IDs dos nós de saída (output_nodes)

        Retornamos (node_order, input_nodes, output_nodes).
        """
        if debug:
            print("Printando 10 primeiras iterações de get_topological_info()...")
            max_iterations = 10
            global iterations
            iterations+=1
            print(f"Iniciando iteration {iterations}...")

        net = FeedForwardNetwork.create(self.genome, self.config)
        
        # 1) 'node_order' começa com os nós de entrada do NEAT
        node_order = list(net.input_nodes)

        # 2) Depois adiciona todos os nós processados em net.node_evals (ordem topológica)
        for node_eval in net.node_evals:
            node_id = node_eval[0]
            node_order.append(node_id)
        
        # 3) Garante que todos os nós de saída (separados) estão em node_order
        for output in net.output_nodes:
            if output not in node_order:
                node_order.append(output)

        
        if debug:
            print(f"\nnet.input_nodes (Retorno 2): {net.input_nodes},"
                f"\nnode_order: {node_order},"
                f"\nnet.node_evals: {net.node_evals},"
                f"\nnode_order(de novo): {node_order},"
                f"\nnet.output_nodes (Retorno 3): {net.output_nodes},"
                f"\nnode_order(Retorno 1): {node_order}")
            
            if (iterations >= max_iterations):
                print(f"\nIterations >= {max_iterations}, saindo...")
                exit(1)

        return node_order, net.input_nodes, net.output_nodes

    def decode_genome_to_torch(self, debug = False):
        """
        Constrói dinamicamente um nn.Module PyTorch
        que respeita a topologia definida pelo NEAT.
        """

        if debug:
            print("Printando 10 primeiras iterações de decode_genome_to_torch()...")
            max_iterations = 10
            global iterations
            iterations+=1
            print(f"Iniciando iteration {iterations}...")

        # Extrai as listas: nós em ordem topológica, nós de entrada, nós de saída
        node_order, in_nodes, out_nodes = self.get_topological_info()

        # Certifica que todos os nós em conexões (genome.connections) 
        # estão no node_order
        for cg in self.genome.connections.values():
            if cg.enabled:
                in_node, out_node = cg.key
                if in_node not in node_order:
                    if debug:
                        print(f"in_node not in node_order, adicionando node: {in_node}...")
                    node_order.append(in_node)
                if out_node not in node_order:
                    if debug:
                        print(f"out_node not in node_order, adicionando node: {out_node}...")
                    node_order.append(out_node)
            



        # Montamos a classe interna NeatModule (nn.Module)
        genome = self.genome
        config = self.config
        topo_order = node_order

        class NeatModule(nn.Module):
            def __init__(_self, genome, config, node_order, in_nodes, out_nodes):
                enabled_count = 0
                super().__init__()
                _self.genome = genome
                _self.config = config
                _self.node_order = node_order
                _self.input_nodes = in_nodes
                _self.output_nodes = out_nodes

                # Coleta conexões habilitadas
                _self.connections = []
                for cg in _self.genome.connections.values():
                    if cg.enabled:
                        enabled_count += 1
                        i_node, o_node = cg.key
                        w = cg.weight
                        _self.connections.append((i_node, o_node, w))
                        if debug:
                            print(f"\nAdicionando conexão ({i_node} -> {o_node}) com peso {w:.3f} em _self.connections...")
                if debug:
                    print(f"\n[NeatModule] genome tem {enabled_count} conexões habilitadas...")


                _self.num_inputs = _self.config.genome_config.num_inputs
                _self.num_outputs = _self.config.genome_config.num_outputs
                
                # Mesma coisa para ver se a order é condizente
                if debug:
                    print(f"\n[NeatModule] node_order len={len(_self.node_order)} "
                    f"(num_inputs={_self.num_inputs}, num_outputs={_self.num_outputs})")

                # Mapeia (node_id) -> índice
                _self.node_id_to_idx = {}
                for i, n_id in enumerate(_self.node_order):
                    _self.node_id_to_idx[n_id] = i
                
                if debug:
                    print(f"\n[NeatModule] node_id_to_idx: {_self.node_id_to_idx}")

            def forward(_self, x):
                """
                1) Alimenta nós de entrada,
                2) Propaga nós intermediários (e nós de saída) com SIGMOID,
                3) Coleta as saídas e faz SOFTMAX no vetor final.
                """
                device = x.device
                batch_size = x.shape[0]
                total_nodes = len(_self.node_order)

                # Cria o tensor de ativações
                activations = x.new_zeros((batch_size, total_nodes))

                if debug:
                    print(f"\n[NeatModule] activations.shape: {activations.shape}")
                    print("Checando se todos são zero...")
                    for i in range(total_nodes):
                        if activations[:, i].sum() != 0:
                            print(f"Erro! activations[:, {i}] não é zero!")
                            exit(1)
                    print("Todos são zero!")

                # (1) Copiar dados de entrada
                for col_index, node_id in enumerate(_self.input_nodes):
                    idx = _self.node_id_to_idx[node_id]
                    activations[:, idx] = x[:, col_index]
                
                if debug:
                    print(f"\n[NeatModule] activations: {activations},"
                          f"x: {x}")

                # (2) Montar adjacency: para cada nó destino, lista de (nó origem, peso)
                adjacency = {n_id: [] for n_id in _self.node_order}

                if debug:
                    print(f"\n[NeatModule] adjacency: {adjacency}")
                for (i_node, o_node, w) in _self.connections:
                    if (i_node not in _self.node_id_to_idx) or (o_node not in _self.node_id_to_idx):
                        continue
                    adjacency[o_node].append((i_node, w))

                # Compute activation for non-input nodes, including bias
                for node_id in _self.node_order:
                    if node_id in _self.input_nodes:
                        continue
                    node_idx = _self.node_id_to_idx[node_id]
                    bias = 0.0
                    if node_id in _self.genome.nodes:
                        bias = _self.genome.nodes[node_id].bias
                    else:
                        if debug:
                            print(f"\n[NeatModule] node_id={node_id} não está em genome.nodes")
                    total_in = torch.full((batch_size,), bias, device=x.device)

                    if debug:
                        print(f"total_in.shape: {total_in.shape}, bias: {bias}, total_in: {total_in}")

                    for (src_id, w) in adjacency[node_id]:
                        src_idx = _self.node_id_to_idx[src_id]
                        total_in += activations[:, src_idx] * w
                    
                    if debug:
                        print(f"\n[NeatModule] total_in (depois de ser somado com activations): total_in.shape: {total_in.shape},"
                              f"total_in: {total_in}")

                    if node_id in _self.output_nodes:
                        # For output nodes, do not apply non-linearity
                        activations[:, node_idx] = total_in
                    else:
                        # Apply SIGMOID activation for hidden nodes
                        activations[:, node_idx] = torch.sigmoid(total_in)

                    if debug:
                        print(f"\n[NeatModule] activations final: {activations}")

                # (4) Extrair vetores de saída e só agora aplicar softmax
                out_vec = []
                for node_id in _self.output_nodes:
                    out_idx = _self.node_id_to_idx[node_id]
                    out_vec.append(activations[:, out_idx])

                if debug:
                    print(f"\n[NeatModule] out_vec: {out_vec}")

                # outs vira uma lista de shape 10, cada item [batch_size]
                # Empilhamos no dim=1 -> shape [batch_size, 10]
                logits = torch.stack(out_vec, dim=1)

                if debug:
                    print(f"\n[NeatModule] logits: {logits}")
                
                # Por fim, (se quisermos) F.softmax:
                #preds = F.softmax(logits, dim=1)

                #if debug:
                #    print(f"\n[NeatModule] preds (Valor retornado pela função forward): {preds}")
                #return preds

                return logits

        ret = NeatModule(genome, config, topo_order, in_nodes, out_nodes)

        if debug:
            print(f"[decode_genome_to_torch]: Retornando NeatModulo(genome, config, topo_order, in_nodes, out_nodes) na função decode_genome_to_torch()..."
                  f"\ncom: genome={genome},"
                  f"\nconfig={config},"
                  f"\ntopo_order={topo_order},"
                  f"\nin_nodes={in_nodes},"
                  f"\nout_nodes={out_nodes}")

            if (iterations >= max_iterations):
                print(f"\nIterations >= {max_iterations}, saindo...")
                exit(1)

        return ret


gen = -1

def eval_genomes(genomes, config, X_train, y_train, debug=False):
    global gen
    
    # Converte os dados para tensores no device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_train_torch = torch.tensor(X_train, dtype=torch.float, device=device)
    y_train_torch = torch.tensor(y_train, dtype=torch.long, device=device)
    
    # Se os rótulos estão one-hot, converte para índices de classe
    if y_train_torch.ndim > 1 and y_train_torch.shape[1] > 1:
        y_train_torch = torch.argmax(y_train_torch, dim=1)
    
    gen += 1
    loss_fn = nn.CrossEntropyLoss()
    
    for genome_id, genome in genomes:
        # Cria a rede PyTorch a partir do genome
        neat_net = Genome(genome, config)
        net = neat_net.decode_genome_to_torch().to(device)
        
        # Forward pass no dataset
        outputs = net(X_train_torch)  # Logits
        
        # Calcula a Cross-Entropy Loss
        loss = loss_fn(outputs, y_train_torch)
        
        # Calcula a fitness como uma função inversa da loss
        fitness = 1 / (1 + loss.item())
        
        if debug:
            print(f"[DEBUG] Gen: {gen} - Genome ID={genome_id} -> Loss={loss.item():.5f}, Fitness={fitness:.5f}")
        
        # Atribui a fitness ao genome
        genome.fitness = fitness
