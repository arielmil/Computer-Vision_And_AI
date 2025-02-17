# Import necessary modules
from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode
from tensorneat.problem.rl import BraxEnv
from tensorneat.common import ACT, AGG
import jax
import jax.numpy as jnp

# Criando uma matriz grande para testar
x = jnp.ones((1000, 1000))

# Verifica onde está rodando
print(x.device())  # Deve retornar algo como: gpu:0

jax.config.update("jax_platform_name", "cuda")  # ou "gpu"


# Define the pipeline
pipeline = Pipeline(
    algorithm=NEAT(
        pop_size=1000,
        species_size=20,
        survival_threshold=0.1,
        compatibility_threshold=1.0,
        genome=DefaultGenome(
            num_inputs=11,
            num_outputs=3,
            init_hidden_layers=(),
            node_gene=BiasNode(
                activation_options=ACT.tanh,
                aggregation_options=AGG.sum,
            ),
            output_transform=ACT.tanh,
        ),
    ),
    problem=BraxEnv(
        env_name="hopper",
        max_step=1000,
    ),
    seed=42,
    generation_limit=100,
    fitness_target=5000,
)

# Initialize state
state = pipeline.setup()

# Run until termination
state, best = pipeline.auto_run(state)