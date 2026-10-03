from collections.abc import Callable, Sequence
from pathlib import Path
import pickle

import neat

INPUT_COUNT = 16
ACTION_COUNT = 4

MOVE_FORWARD = 1
MOVE_BACKWARD = 2
ROTATE_LEFT = 3
ROTATE_RIGHT = 4


def action_from_features(
    network: neat.nn.FeedForwardNetwork, features: Sequence[float]
) -> int:
    """Return 1=forward, 2=backward, 3=left, or 4=right."""
    if len(features) != INPUT_COUNT:
        raise ValueError(
            f"Expected {INPUT_COUNT} CNN features, got {len(features)}"
        )

    outputs = network.activate(features)
    if len(outputs) != ACTION_COUNT:
        raise ValueError(
            f"Expected {ACTION_COUNT} action outputs, got {len(outputs)}"
        )
    return max(range(ACTION_COUNT), key=outputs.__getitem__) + 1


def _evaluate_genomes(genomes, config, evaluate_episode: Callable) -> None:
    for _, genome in genomes:
        network = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = evaluate_episode(network)


class NeatTrainer:
    def __init__(self) -> None:
        config_path = Path(__file__).with_name("neat-config.ini")
        self.checkpoint_dir = Path(__file__).resolve().parents[1] / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.best_checkpoint_path = self.checkpoint_dir / "best-checkpoint.pkl"
        self.last_checkpoint_path = self.checkpoint_dir / "last-checkpoint"
        self.legacy_brain_path = self.checkpoint_dir / "best-brain.pkl"
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(config_path),
        )
        self.best_genome = None
        self.best_genome_needs_revalidation = False
        legacy_checkpoints = list(self.checkpoint_dir.glob("neat-checkpoint-*"))
        if self.last_checkpoint_path.exists():
            self.population = neat.Checkpointer.restore_checkpoint(
                str(self.last_checkpoint_path)
            )
        elif legacy_checkpoints:
            latest_checkpoint = max(
                legacy_checkpoints,
                key=lambda path: int(path.name.rsplit("-", 1)[-1]),
            )
            self.population = neat.Checkpointer.restore_checkpoint(
                str(latest_checkpoint)
            )
        else:
            self.population = neat.Population(self.config)

        self.checkpointer = neat.Checkpointer(
            generation_interval=None,
            filename_prefix=str(self.checkpoint_dir / "last-checkpoint-"),
        )
        self.checkpointer.last_generation_checkpoint = self.population.generation
        self._add_reporters()
        self._load_best_genome()
        if not self.last_checkpoint_path.exists() and legacy_checkpoints:
            self.save_checkpoint()
        elif self.best_genome is not None and not self.best_checkpoint_path.exists():
            self._save_best_checkpoint()
        self._remove_legacy_checkpoints()

    def _add_reporters(self) -> None:
        self.population.add_reporter(neat.StdOutReporter(True))
        self.population.add_reporter(neat.StatisticsReporter())

    def evolve_generation(self, evaluate_episode: Callable) -> None:
        generation_best = self.population.run(
            lambda genomes, config: _evaluate_genomes(
                genomes, config, evaluate_episode
            ),
            1,
        )
        if generation_best is not None and (
            self.best_genome is None
            or self.best_genome_needs_revalidation
            or generation_best.fitness > self.best_genome.fitness
        ):
            self.best_genome = generation_best
            self.best_genome_needs_revalidation = False
            self._save_best_checkpoint()
        if self.population.generation % 5 == 0:
            self.save_checkpoint()

    def seed_population_with_best(self) -> bool:
        self._load_best_genome()
        if self.best_genome is None or not self.population.population:
            return False

        weakest_genome = min(
            self.population.population.values(),
            key=lambda genome: (
                genome.fitness if genome.fitness is not None else float("-inf")
            ),
        )
        seed = pickle.loads(pickle.dumps(self.best_genome))
        seed.key = weakest_genome.key
        seed.fitness = None
        self.population.population[seed.key] = seed

        for species in self.population.species.species.values():
            if weakest_genome.key in species.members:
                species.members[seed.key] = seed
                break

        return True

    def save_checkpoint(self) -> None:
        for temporary_checkpoint in self.checkpoint_dir.glob("last-checkpoint-*"):
            temporary_checkpoint.unlink()
        self.checkpointer.save_checkpoint(
            self.config,
            self.population.population,
            self.population.species,
            self.population.generation,
        )
        generated_checkpoint = self.checkpoint_dir / (
            f"last-checkpoint-{self.population.generation}"
        )
        generated_checkpoint.replace(self.last_checkpoint_path)
        self.checkpointer.last_generation_checkpoint = self.population.generation
        if self.best_genome is not None:
            self._save_best_checkpoint()
        self._remove_legacy_checkpoints()

    def restore_latest_checkpoint(self):
        if not self.last_checkpoint_path.exists():
            return None

        self.population = neat.Checkpointer.restore_checkpoint(
            str(self.last_checkpoint_path)
        )
        self.checkpointer.last_generation_checkpoint = self.population.generation
        self._add_reporters()
        self._load_best_genome()
        return self.load_best_network()

    def load_best_network(self):
        self._load_best_genome()
        if self.best_genome is None and self.population.best_genome is not None:
            self.best_genome = self.population.best_genome
            self.best_genome_needs_revalidation = False

        best_genome = self.best_genome
        if best_genome is None:
            return None
        return neat.nn.FeedForwardNetwork.create(best_genome, self.config)

    def _load_best_genome(self) -> None:
        brain_path = (
            self.best_checkpoint_path
            if self.best_checkpoint_path.exists()
            else self.legacy_brain_path
        )
        if brain_path.exists():
            with brain_path.open("rb") as brain_file:
                self.best_genome = pickle.load(brain_file)
            self.best_genome_needs_revalidation = self.best_genome.fitness == 0.0

    def _save_best_checkpoint(self) -> None:
        if self.best_genome is None:
            return
        temporary_path = self.best_checkpoint_path.with_suffix(".pkl.tmp")
        with temporary_path.open("wb") as checkpoint_file:
            pickle.dump(self.best_genome, checkpoint_file)
        temporary_path.replace(self.best_checkpoint_path)
        if self.legacy_brain_path.exists():
            self.legacy_brain_path.unlink()

    def _remove_legacy_checkpoints(self) -> None:
        for old_checkpoint in self.checkpoint_dir.glob("neat-checkpoint-*"):
            old_checkpoint.unlink()
        if self.legacy_brain_path.exists():
            self.legacy_brain_path.unlink()
