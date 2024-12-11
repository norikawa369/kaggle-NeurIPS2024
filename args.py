# chemprop/args
import json
import os
from tempfile import TemporaryDirectory
import pickle
from typing import List, Optional, Tuple
from typing_extensions import Literal

import torch
from tap import Tap # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)



Metric = Literal['auc', 'prc-auc', 'rmse', 'mae', 'mse', 'r2', 'accuracy', 'cross_entropy', 'binary_cross_entropy', 'bceloss', 'APC']


def get_checkpoint_paths(checkpoint_path: Optional[str] = None,
                         checkpoint_paths: Optional[List[str]] = None,
                         checkpoint_dir: Optional[str] = None,
                         ext: str = '.pt') -> Optional[List[str]]:
    """
    Gets a list of checkpoint paths either from a single checkpoint path or from a directory of checkpoints.
    If :code:`checkpoint_path` is provided, only collects that one checkpoint.
    If :code:`checkpoint_paths` is provided, collects all of the provided checkpoints.
    If :code:`checkpoint_dir` is provided, walks the directory and collects all checkpoints.
    A checkpoint is any file ending in the extension ext.
    :param checkpoint_path: Path to a checkpoint.
    :param checkpoint_paths: List of paths to checkpoints.
    :param checkpoint_dir: Path to a directory containing checkpoints.
    :param ext: The extension which defines a checkpoint file.
    :return: A list of paths to checkpoints or None if no checkpoint path(s)/dir are provided.
    """
    if sum(var is not None for var in [checkpoint_dir, checkpoint_path, checkpoint_paths]) > 1:
        raise ValueError('Can only specify one of checkpoint_dir, checkpoint_path, and checkpoint_paths')

    if checkpoint_path is not None:
        return [checkpoint_path]

    if checkpoint_paths is not None:
        return checkpoint_paths

    if checkpoint_dir is not None:
        checkpoint_paths = []

        for root, _, files in os.walk(checkpoint_dir):
            for fname in files:
                if fname.endswith(ext):
                    checkpoint_paths.append(os.path.join(root, fname))

        if len(checkpoint_paths) == 0:
            raise ValueError(f'Failed to find any checkpoints with extension "{ext}" in directory "{checkpoint_dir}"')

        return checkpoint_paths

    return None


class CommonArgs(Tap):
    """:class:`CommonArgs` contains arguments that are used in both :class:`TrainArgs` and :class:`PredictArgs`."""

    checkpoint_dir: str = None
    """Directory from which to load model checkpoints (walks directory and ensembles all models that are found)."""
    checkpoint_path: str = None
    """Path to model checkpoint (:code:`.pt` file)."""
    checkpoint_paths: List[str] = None
    """List of paths to model checkpoints (:code:`.pt` files)."""
    no_cuda: bool = False
    """Turn off cuda (i.e., use CPU instead of GPU)."""
    gpu: int = None
    """Which GPU to use."""
    max_data_size: int = None
    """Maximum number of data points to load."""
    num_workers: int = 12
    """Number of workers for the parallel data loading (0 means sequential)."""
    batch_size: int = 16
    """Batch size."""
    iters_accumulate: int = 1
    """iters to loss accumulate"""
    
    def __init__(self, *args, **kwargs):
        super(CommonArgs, self).__init__(*args, **kwargs)
        self._atom_features_size = 0
        self._bond_features_size = 0
        self._atom_descriptors_size = 0

    @property
    def device(self) -> torch.device:
        """The :code:`torch.device` on which to load and process data and models."""
        if not self.cuda:
            return torch.device('cpu')

        return torch.device('cuda', self.gpu)

    @device.setter
    def device(self, device: torch.device) -> None:
        self.cuda = device.type == 'cuda'
        self.gpu = device.index

    @property
    def cuda(self) -> bool:
        """Whether to use CUDA (i.e., GPUs) or not."""
        return not self.no_cuda and torch.cuda.is_available()

    @cuda.setter
    def cuda(self, cuda: bool) -> None:
        self.no_cuda = not cuda

    def configure(self) -> None:
        self.add_argument('--gpu', choices=list(range(torch.cuda.device_count())))
        # self.add_argument('--features_generator', choices=get_available_features_generators())

    def process_args(self) -> None:
        # Load checkpoint paths
        self.checkpoint_paths = get_checkpoint_paths(
            checkpoint_path=self.checkpoint_path,
            checkpoint_paths=self.checkpoint_paths,
            checkpoint_dir=self.checkpoint_dir,
        )


class TrainArgs(CommonArgs):
    """:class:`TrainArgs` includes :class:`CommonArgs` along with additional arguments used for training a Chemprop model."""

    # General arguments
    data_dir: str = '/work/ge58/e58004/kaggle-comps/kaggle-leashbio-belka/input/shrunken_data/'
    # path of smiles-graph dict pickle file
    graph_dir: str = '/work/ge58/e58004/kaggle-comps/kaggle-leashbio-belka/input/custom_data/'
    # data extention
    ext: str = 'parquet'
    # protein name to train
    proteins: List = ['BRD4', 'HSA', 'sEH']
    
    """Method of splitting the data into train/val/test."""
    num_folds: int = 8
    """Number of folds when performing cross validation."""
    # folds_file: str = None
    """Optional file of fold labels."""
    seed: int = 0
    """
    Random seed to use when splitting data into train/val/test sets.
    When :code`num_folds > 1`, the first fold uses this seed and all subsequent folds add 1 to the seed.
    """
    pytorch_seed: int = 0
    """Seed for PyTorch randomness (e.g., random initial weights)."""
    metric: Metric = 'bceloss'
    """
    Metric to use during evaluation. It is also used with the validation set for early stopping.
    Defaults to "auc" for classification and "rmse" for regression.
    """
    extra_metrics: List[Metric] = []
    
    # train_folds: List[int] = [0,1,2,3,4,5,6,7]
    # train_folds: List[int] = [0,1,2,3]
    train_folds: List[int] = [0]
    """folds to train"""

    save_dir: str = '/home/akawa005/kaggle-comps/kaggle-leashbio-belka/'
    """Directory where model checkpoints will be saved."""
    
    debug_mode: bool = False
    """debug_mode uses low memory by cut data."""

    data_extension: Literal['csv', 'pkl', 'parquet'] = 'parquet'
    
    checkpoint_frzn: str = None
    """Path to model checkpoint file to be loaded for overwriting and freezing weights."""
    test: bool = False
    """Whether to skip training and only test the model."""
    quiet: bool = False
    """Skip non-essential print statements."""
    log_frequency: int = 10
    """The number of batches between each logging of the training loss."""
    show_individual_scores: bool = False
    """Show all scores for individual targets, not just average, at the end."""
    cache_cutoff: float = 10000
    """
    Maximum number of molecules in dataset to allow caching.
    Below this number, caching is used and data loading is sequential.
    Above this number, caching is not used and data loading is parallel.
    Use "inf" to always cache.
    """
    save_preds: bool = False
    """Whether to save test split predictions during training."""
    resume_experiment: bool = False
    """
    Whether to resume the experiment.
    Loads test results from any folds that have already been completed and skips training those folds.
    """

    # threshold of between positive and negative
    # new!!
    threshold: float = 6.5

    # Model arguments
    bias: bool = False
    """Whether to add bias to linear layers."""
    # hidden_size: int = 300

    # MPNN parameters
    graph_dims: int = 96
    """
    PACK_NODE_DIM=9
    PACK_EDGE_DIM=1
    NODE_DIM=PACK_NODE_DIM*8
    EDGE_DIM=PACK_EDGE_DIM*8
    """
    node_dim: int = 72
    edge_dim: int = 8

    num_layers: int = 4



    atom_dim: int= 34
    n_layers_EAA: int = 1
    n_layers_PD: int = 1
    n_layers_D: int = 1
    n_layers_gru: int = 1
    # hid_dim: int = 768
    hid_dim: int = 64
    hid_dim_fc: int = 256

    n_head: int = 8



    """Dimensionality of hidden layers in MPN."""
    # depth: int = 3
    """Number of message passing steps."""
    # mpn_shared: bool = False
    """Whether to use the same message passing neural network for all input molecules
    Only relevant if :code:`number_of_molecules > 1`"""
    dropout: float = 0.1
    """Dropout probability."""
    config_path: str = None
    """
    Path to a :code:`.json` file containing arguments. Any arguments present in the config file
    will override arguments specified via the command line or by the defaults.
    """
    # ensemble_size: int = 1
    """Number of models in ensemble."""
    # aggregation: Literal['mean', 'sum', 'norm'] = 'mean'
    """Aggregation scheme for atomic vectors into molecular vectors"""
    # aggregation_norm: int = 100
    """For norm aggregation, number by which to divide summed up atomic features"""
    # reaction: bool = False
    """
    Whether to adjust MPNN layer to take reactions as input instead of molecules.
    """
    # reaction_mode: Literal['reac_prod', 'reac_diff', 'prod_diff'] = 'reac_diff'
    """
    Choices for construction of atom and bond features for reactions
    :code:`reac_prod`: concatenates the reactants feature with the products feature.
    :code:`reac_diff`: concatenates the reactants feature with the difference in features between reactants and products.
    :code:`prod_diff`: concatenates the products feature with the difference in features between reactants and products.
    """
    # explicit_h: bool = False
    """
    Whether H are explicitly specified in input (and should be kept this way).
    """

    # Training arguments
    epochs: int = 2
    """Number of epochs to run."""
    # warmup_epochs: float = 2.0
    """
    Number of epochs during which learning rate increases linearly from :code:`init_lr` to :code:`max_lr`.
    Afterwards, learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr`.
    """
    init_lr: float = 1e-4
    """Initial learning rate."""

    decay_interval: float = 5
    lr_decay: float = 1.0
    weight_decay: float = 1e-4


    # max_lr: float = 1e-3
    """Maximum learning rate."""
    # final_lr: float = 1e-4
    """Final learning rate."""
    grad_clip: float = None
    """Maximum magnitude of gradient during training."""
    class_balance: bool = False
    """Trains with an equal number of positives and negatives in each batch."""



    def __init__(self, *args, **kwargs) -> None:
        super(TrainArgs, self).__init__(*args, **kwargs)
        self._task_names = None
        self._crossval_index_sets = None
        self._task_names = None
        self._num_tasks = None
        self._features_size = None
        self._train_data_size = None

    @property
    def metrics(self) -> List[str]:
        """The list of metrics used for evaluation. Only the first is used for early stopping."""
        return [self.metric] + self.extra_metrics

    @property
    def minimize_score(self) -> bool:
        """Whether the model should try to minimize the score metric or maximize it."""
        return self.metric in {'rmse', 'mae', 'mse', 'cross_entropy', 'binary_cross_entropy'}

    @property
    def num_lrs(self) -> int:
        """The number of learning rates to use (currently hard-coded to 1)."""
        return 1

    @property
    def crossval_index_sets(self) -> List[List[List[int]]]:
        """Index sets used for splitting data into train/validation/test during cross-validation"""
        return self._crossval_index_sets

    @property
    def task_names(self) -> List[str]:
        """A list of names of the tasks being trained on."""
        return self._task_names

    @task_names.setter
    def task_names(self, task_names: List[str]) -> None:
        self._task_names = task_names

    @property
    def num_tasks(self) -> int:
        """The number of tasks being trained on."""
        return len(self.task_names) if self.task_names is not None else 0

    @property
    def features_size(self) -> int:
        """The dimensionality of the additional molecule-level features."""
        return self._features_size

    @features_size.setter
    def features_size(self, features_size: int) -> None:
        self._features_size = features_size

    @property
    def train_data_size(self) -> int:
        """The size of the training data set."""
        return self._train_data_size

    @train_data_size.setter
    def train_data_size(self, train_data_size: int) -> None:
        self._train_data_size = train_data_size


    def process_args(self) -> None:
        # super().process_args()はコマンドラインの時には実行する
        super(TrainArgs, self).process_args()

        global temp_dir  # Prevents the temporary directory from being deleted upon function return

        # Load config file
        if self.config_path is not None:
            with open(self.config_path) as f:
                config = json.load(f)
                for key, value in config.items():
                    setattr(self, key, value)

        # Create temporary directory as save directory if not provided
        if self.save_dir is None:
            temp_dir = TemporaryDirectory()
            self.save_dir = temp_dir.name


        # Test settings
        if self.test:
            self.epochs = 0
