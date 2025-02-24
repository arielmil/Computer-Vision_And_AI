import torch
import torch.nn as nn
import torch.nn.functional as F
from neat.nn import FeedForwardNetwork

def get_topological_order(genome, config):
    # Cria a rede feedforward (padrão da lib) só para obter a ordem
    net = FeedForwardNetwork.create(genome, config)
    # O objeto net tem um atributo "node_evals" (lista de tuplas) 
    # que é processado na ordem topológica.
    
    # net.input_nodes e net.output_nodes também estão disponíveis.
    # Por exemplo:
    #
    # net.input_nodes  -> [0, 1, 2, ...] IDs dos nós de entrada
    # net.output_nodes -> [X, Y, Z, ...] IDs dos nós de saída
    #
    # net.node_evals é uma lista de namedtuples (or tuples) do tipo:
    #   (node, activation, agg_function, bias, links)
    # e esse "node" está em ordem topológica.
    
    ordered_nodes = list(net.input_nodes)  # começa com inputs
    
    # Agora percorremos "node_evals" na ordem:
    for node_eval in net.node_evals:
        node_id = node_eval[0]  # o primeiro item é o ID do nó
        ordered_nodes.append(node_id)
    
    return ordered_nodes

def decode_genome_to_torch(genome, config):
    # Parecido com a ideia do JAX,
    # mas vamos montar um "nn.Module" manualmente.
    # Observando que a topologia é arbitrária.

    class NeatModule(nn.Module):
        def __init__(self):
            super().__init__()
            # Precisaríamos armazenar no self
            # as estruturas de conexões, e possivelmente
            # converter os pesos para Tensors
            self.connections = []
            for cg in genome.connections.values():
                if cg.enabled:
                    in_node, out_node = cg.key
                    w = cg.weight
                    self.connections.append((in_node, out_node, w))

            self.num_inputs = config.genome_config.num_inputs
            self.num_outputs = config.genome_config.num_outputs
            self.node_order = get_topological_order(genome, config)
            self.node_id_to_idx = {}
            for i, n_id in enumerate(self.node_order):
                self.node_id_to_idx[n_id] = i

        def forward(self, x):
            # x shape = [batch_size, num_inputs]
            batch_size = x.shape[0]
            total_nodes = len(self.node_order)
            # Cria as ativações
            activations = x.new_zeros((batch_size, total_nodes))
            # Copia as inputs:
            for i_in in range(self.num_inputs):
                idx = self.node_id_to_idx[i_in]
                activations[:, idx] = x[:, i_in]

            adjacency = {}
            for n_id in self.node_order:
                adjacency[n_id] = []
            for (i_node, o_node, w) in self.connections:
                adjacency[o_node].append((i_node, w))

            # Calcula na ordem topológica
            for node_id in self.node_order[self.num_inputs:]:
                in_info = adjacency[node_id]
                node_idx = self.node_id_to_idx[node_id]
                total_in = torch.zeros(batch_size, device=x.device)
                for (src_id, w) in in_info:
                    src_idx = self.node_id_to_idx[src_id]
                    total_in += activations[:, src_idx] * w
                # Exemplo: ReLU
                node_out = F.relu(total_in)
                activations[:, node_idx] = node_out

            # Extrair outputs
            outs = []
            for i_out in range(self.num_outputs):
                out_node_id = self.num_inputs + i_out
                out_idx = self.node_id_to_idx[out_node_id]
                outs.append(activations[:, out_idx])
            return torch.stack(outs, dim=1)

    return NeatModule()

# E então, no eval_genomes, você faz:
def eval_genomes(genomes, config):
    X_train_torch = torch.tensor(X_train, dtype=torch.float, device='cuda')
    y_train_torch = torch.tensor(y_train, dtype=torch.long, device='cuda')

    for genome_id, genome in genomes:
        net = decode_genome_to_torch(genome, config).to('cuda')
        outputs = net(X_train_torch)  # [batch_size, num_outputs]

        # se quiser crossentropy,
        # PyTorch tem F.cross_entropy que aceita logits
        # mas, nesse caso, outputs = ReLU. Precisaria ou de no final linear,
        # ou de um linear + no transform, etc.

        # Exemplo: cross-entropy "manual"
        log_probs = F.log_softmax(outputs, dim=1)
        loss = -log_probs[range(len(y_train_torch)), y_train_torch].mean()

        genome.fitness = float(-loss.detach().cpu().numpy())
