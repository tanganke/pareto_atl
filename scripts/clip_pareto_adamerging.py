from _common import *

log = logging.getLogger(__name__)

from collections import defaultdict

from fusionlib.merge.adamerging import entropy_loss
from fusionlib.merge.wrapper.layer_wise_fusion import (
    LayerWiseMergedModel,
    get_layer_wise_weights,
)
from fusionlib.utils import first, timer
from fusionlib.utils.torch.state_dict_arithmetic import state_dict_sub
from fusionlib.utils.torch.state_dict_arithmetic import (
    to_device as state_dict_to_device,
)

from scripts.clip_program import CLIPProgram
from src.data.clip_datasets.common import maybe_dictionarize
from src.min_norm_solvers import MinNormSolver


class CLIPParetoAdaMerging(CLIPProgram):
    def main(self):
        cfg = self.cfg

        self.load_clip_models()
        self.load_datasets()

        if cfg.pareto_tta:
            self.build_model()
            self.pareto_tta()
        if cfg.evaluate:
            self.evaluate()

    def build_model(self):
        cfg = self.cfg

        with timer("calculate task vectors"):
            task_vectors = {}
            for model_name in self.finetuned_models:
                task_vectors[model_name] = state_dict_sub(
                    self.finetuned_models[model_name].state_dict(),
                    self.pretrained_model.state_dict(),
                )
                task_vectors[model_name] = state_dict_to_device(
                    task_vectors[model_name], "cuda"
                )
        adamerging_weights = get_layer_wise_weights(
            num_models=len(self.finetuned_models),
            num_layers=len(first(task_vectors.values())),
            init_values=cfg.init_lambda,
        ).cuda()
        model = LayerWiseMergedModel(
            deepcopy(self.pretrained_model).requires_grad_(False).cuda(),
            layer_wise_weights=adamerging_weights,
            task_vectors=list(task_vectors.values()),
            clamp_weights=False,
        )

        self.model = model

    def pareto_tta(self):
        cfg = self.cfg

        adamerging_weights = self.model.layer_wise_weights
        model = self.model
        optimizer = torch.optim.Adam((adamerging_weights,), lr=cfg.lr)

        model.train()
        for step_idx in tqdm(range(1, 1001), "Pareto TTA", dynamic_ncols=True):

            model.merge_weights()
            all_grads = []
            for task_name in cfg.seen_datasets:
                batch = next(self.shuffled_test_loader_iters[task_name])
                batch = maybe_dictionarize(batch)
                # labels are not used during TTA
                images = batch["images"].cuda()

                features = model(images)
                logits = self.classification_heads[task_name](features)

                loss = entropy_loss(logits=logits, reduction="mean")

                grad = (
                    torch.autograd.grad(
                        loss, adamerging_weights, create_graph=False, retain_graph=True
                    )[0]
                    .detach_()
                    .flatten()
                )
                all_grads.append(grad)

            sol, min_norm = MinNormSolver.find_min_norm_element(all_grads)
            if not isinstance(sol, torch.Tensor):
                sol = torch.from_numpy(sol)
            sol = sol.to(
                device=adamerging_weights.device, dtype=adamerging_weights.dtype
            )
            for i in range(len(sol)):
                self.writer.add_scalar(
                    f"pareto_tta/lambda_{i}", sol[i].item(), step_idx
                )

            optimizer.zero_grad()
            grad = torch.stack(all_grads) * sol.view(-1, 1)
            adamerging_weights.grad = grad.sum(dim=0).view_as(adamerging_weights)
            optimizer.step()

            if cfg.fast_dev_run:
                print("fast_dev_run, skip the rest")
                break

            if (step_idx) % 100 == 0:
                log.info(f"save checkpoint at step {step_idx}")
                torch.save(
                    {
                        "state_dict": state_dict_to_device(
                            model.merged_state_dict, device="cpu", inplace=False
                        ),
                        "layer_wise_weights": adamerging_weights.to("cpu", copy=True),
                    },
                    os.path.join(self.checkpoint_dir, f"checkpoint-{step_idx}.ckpt"),
                )

    @torch.no_grad()
    def evaluate(self):
        cfg = self.cfg
        checkpoint_dir = (
            cfg.checkpoint_dir
            if cfg.checkpoint_dir is not None
            else self.checkpoint_dir
        )

        results = defaultdict(list)
        for step_idx in tqdm(range(100, 1001, 100), "evaluate"):
            model = deepcopy(self.pretrained_model)
            ckpt = torch.load(
                os.path.join(checkpoint_dir, f"checkpoint-{step_idx}.ckpt"),
                map_location="cpu",
            )
            model.load_state_dict(ckpt["state_dict"])
            model.cuda()
            model.eval()
            results["step"].append(step_idx)
            for dataset_name in self.test_datasets:
                results[dataset_name].append(
                    self.evaluate_model(
                        model,
                        self.classification_heads[dataset_name],
                        self.test_loaders[dataset_name],
                        device="cuda",
                    )
                )

            df = pd.DataFrame(results)
            print(df)
            df.to_csv(os.path.join(checkpoint_dir, "results.csv"), index=False)

        log.info(df)


@hydra.main(
    config_path=str(CONFIG_DIR),
    config_name="clip_pareto_adamerging",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    program = CLIPParetoAdaMerging(cfg)
    program.main()


if __name__ == "__main__":
    main()
