from _common import *

log = logging.getLogger(__name__)

from typing import Optional

from fusionlib.utils import timer
from fusionlib.utils.hydra import HydraProgram
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.clip_datasets.common import maybe_dictionarize
from src.heads import ClassificationHead, get_classification_head
from src.modeling import ImageEncoder

CHECKPOINT_DIR = CACHE_DIR / "models" / "checkpoints" / "task_vectors_checkpoints"
MODELS = ["ViT-B-16", "ViT-B-32", "ViT-L-14"]
DATASETS = ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"]


def pretrained_model_path(model_name: str) -> Path:
    """
    This function generates the path for the pretrained model.

    Parameters:
        model_name (str): The name of the pretrained model.

    Returns:
        Path: The path of the pretrained model.
    """
    if model_name not in MODELS:
        log.warning(f"Unknown model {model_name}")
    path = CHECKPOINT_DIR / model_name / "zeroshot.pt"
    assert path.is_file(), f"Pretrained model not found at {path}"
    return path


def finetuned_model_path(model_name: str, dataset_name: str) -> Path:
    """
    This function generates the path for the fine-tuned model.

    Parameters:
        model_name (str): The name of the model.
        dataset_name (str): The name of the dataset.

    Returns:
        Path: the path of the fine-tuned model.
    """
    if model_name not in MODELS:
        log.warning(f"Unknown model {model_name}")
    if dataset_name not in DATASETS:
        log.warning(f"Unknown dataset {dataset_name}")
    path = CHECKPOINT_DIR / model_name / dataset_name / "finetuned.pt"
    assert path.is_file(), f"Finetuned model not found at {path}"
    return path


class CLIPProgram(HydraProgram):
    _writer: Optional[SummaryWriter] = None

    def __init__(self, cfg: DictConfig):
        cfg.save = str(CHECKPOINT_DIR / cfg.model)
        cfg.data_location = str(CACHE_DIR / "clip_datasets")
        if cfg.fast_dev_run:
            cfg.num_workers = 0

        super().__init__(cfg)

        log.info(f"working directory: {self.working_dir}")
        log.info(f"output directory: {self.output_dir}")

        self._checkpoint_dir = None

    @property
    def checkpoint_dir(self):
        if self._checkpoint_dir is None:
            self._checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        dir = self._checkpoint_dir
        os.makedirs(dir, exist_ok=True)
        return dir

    @checkpoint_dir.setter
    def checkpoint_dir(self, value):
        self._checkpoint_dir = value

    def initialize_tb_writer(self, comment: str = "tb_log"):
        self._writer = SummaryWriter(log_dir=os.path.join(self.output_dir, comment))

    @property
    def writer(self):
        if self._writer is None:
            self.initialize_tb_writer()
        return self._writer

    def load_clip_models(self):
        """
        This method loads the pretrained and finetuned models for the CLIP (Contrastive Language–Image Pretraining) model.
        It also sets up the classification heads for the test datasets.

        After this method is called, the pretrained model, finetuned models, and classification heads can be accessed as attributes of the instance.
        """
        cfg = self.cfg

        with timer("load clip models"):
            print("loading pre-trained model")
            pretrained_model = torch.load(
                pretrained_model_path(cfg.model_name), map_location="cpu"
            )
            finetuned_models = {}
            for dataset_name in tqdm(cfg.seen_datasets, "loading finetuned models"):
                log.info(f"loading fine-tuned model for {dataset_name}")
                finetuned_models[dataset_name] = torch.load(
                    finetuned_model_path(cfg.model_name, dataset_name),
                    map_location="cpu",
                )

        with timer("load classification heads"):
            classification_heads = {
                dataset_name: get_classification_head(cfg, dataset_name).eval()
                for dataset_name in cfg.test_datasets
            }

        self.pretrained_model: ImageEncoder = pretrained_model
        self.finetuned_models: Dict[str, ImageEncoder] = finetuned_models
        self.classification_heads: Dict[str, ClassificationHead] = classification_heads
        self.classification_heads = {
            task_name: heads.cuda()
            for task_name, heads in self.classification_heads.items()
        }

    def load_datasets(self):
        """
        Loads the datasets specified in the configuration.

        It first imports the necessary modules and sets up a basic transform for the images.
        It then loads each dataset specified in the configuration, applies the basic transform,
        and sets the location, batch size, and number of workers from the configuration.

        The test dataset from each loaded dataset is added to the list of test datasets.
        It then sets up the data loaders for the test datasets, both with
        and without shuffling, and creates an iterator for each shuffled test loader.

        Side Effects:
            Sets the instance variables `test_datasets`, `test_loaders`, `shuffled_test_loaders`, and
            `shuffled_test_loader_iters`.
        """
        if (self.cfg.corruption is None) or (self.cfg.corruption == "clean"):
            from src.data.clip_datasets.registry import get_dataset
        else:
            from src.data.clip_datasets.corruption.registry import get_dataset

        cfg = self.cfg

        dataset_kwargs = dict(
            location=cfg.data_location,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
        if (self.cfg.corruption is not None) and (self.cfg.corruption != "clean"):
            dataset_kwargs["corruption"] = self.cfg.corruption
        datasets = {
            dataset_name: get_dataset(
                dataset_name, self.pretrained_model.val_preprocess, **dataset_kwargs
            )
            for dataset_name in cfg.test_datasets
        }
        self.test_datasets = {
            task_name: dataset.test_dataset for task_name, dataset in datasets.items()
        }
        self.test_loaders = {
            task_name: DataLoader(
                dataset,
                shuffle=False,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                pin_memory=False,
            )
            for task_name, dataset in self.test_datasets.items()
        }
        self.shuffled_test_loaders = {
            task_name: DataLoader(
                dataset,
                shuffle=True,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                pin_memory=False,
            )
            for task_name, dataset in self.test_datasets.items()
        }
        self.shuffled_test_loader_iters = {
            task_name: iter(itertools.cycle(dataloader))
            for task_name, dataloader in self.shuffled_test_loaders.items()
        }

    @torch.no_grad()
    def evaluate_model(
        self,
        model: ImageEncoder,
        classification_head: ClassificationHead,
        dataloader: DataLoader,
        device="cuda",
    ):
        model.eval()
        classification_head.eval()
        correct = 0
        total = 0
        for batch in (pbar := tqdm(dataloader, dynamic_ncols=True, leave=False)):
            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            logits = classification_head(model(images))
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_description(f"accuracy: {correct / total:.2%}")
        return correct / total

    def __del__(self):
        if self._writer:
            self._writer.close()


@hydra.main(
    config_path=str(CONFIG_DIR),
    config_name="clip_default",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    program = CLIPProgram(cfg)
    program.load_clip_models()
    program.load_datasets()


if __name__ == "__main__":
    main()
