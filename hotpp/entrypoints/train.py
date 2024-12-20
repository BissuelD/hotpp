import logging, time, yaml, os, shutil
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel
from ase.data import atomic_numbers
from ..utils import setup_seed, expand_para, EnvPara
from ..model import (
    MiaoNet,
    MiaoMiaoNet,
    LitAtomicModule,
    MultiAtomicModule,
    TwoBody,
    SpinMiaoNet,
    GroundEnergy,
)
from ..layer.cutoff import *
from ..layer.embedding import AtomicEmbedding
from ..layer.radial import *
from ..data import LitAtomsDataset


# 别管Warning不Warning，只要能跑不就行
import warnings

warnings.filterwarnings(action='ignore', message='Checkpoint directory')
warnings.filterwarnings(action='ignore', message='Mean of empty slice')
warnings.filterwarnings(action='ignore', message='The dirpath has changed from')
warnings.filterwarnings(
    action='ignore', message='invalid value encountered in double_scalars'
)

# torch.set_float32_matmul_precision("high")
log = logging.getLogger(__name__)

DefaultPara = {
    "workDir": os.getcwd(),
    "seed": np.random.randint(0, 100000000),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "precision": "float",
    "Data": {
        "path": os.getcwd(),
        "trainBatch": 32,
        "testBatch": 32,
        "std": "force",
        "mean": None,
        "groundEnergy": None,
        "nNeighbor": None,
        "elements": None,
        "numWorkers": 0,
        "pinMemory": False,
        "batchType": "structure",
        "meta": None,
    },
    "Model": {
        "net": "miao",
        "convMode": "node_j",
        "eleMode": "node_edge",
        "updateEdge": "no_update",
        "activateFn": "silu",
        "nEmbedding": 64,
        "nLayer": 5,
        "maxRWay": 2,
        "maxMWay": 2,
        "maxOutWay": 2,
        "maxNBody": 3,
        "nHidden": 64,
        "targetWay": {0: 'site_energy'},
        "CutoffLayer": {
            "type": "poly",
            "p": 5,
        },
        "RadialLayer": {
            "type": "besselMLP",
            "nBasis": 8,
            "nHidden": [64, 64, 64],
            "activateFn": "silu",
        },
        "Repulsion": 0,
        "Spin": False,
    },
    "Train": {
        "maxEpoch": 10000,
        "maxStep": 1000000,
        "allowMissing": False,
        "huberDelta": -1.0,
        "targetProp": ["energy", "forces"],
        "weight": [0.1, 1.0],
        "forceScale": 0.0,
        "evalStepInterval": 50,
        "evalEpochInterval": 1,
        "logInterval": 50,
        "saveStart": 1000,
        "evalTest": True,
        "gradClip": None,
        "Optimizer": {
            "type": "Adam",
            "amsGrad": True,
            "weightDecay": 0.0,
            "learningRate": 0.01,
        },
        "LrScheduler": {
            "type": "constant",
        },
        "emaDecay": 0.0,
    },
}


class SaveModelCheckpoint(ModelCheckpoint):
    """
    Saves model.pt for eval
    """

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        dirname = os.path.dirname(filepath)
        modelname = os.path.basename(filepath)[:-5]
        if trainer.is_global_zero:
            torch.save(
                trainer.lightning_module.model, os.path.join(dirname, f"{modelname}.pt")
            )
            shutil.copy(
                os.path.join(dirname, f"{modelname}.ckpt"),
                os.path.join(dirname, "best.ckpt"),
            )
            shutil.copy(
                os.path.join(dirname, f"{modelname}.pt"),
                os.path.join(dirname, "best.pt"),
            )

    def _remove_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        super()._remove_checkpoint(trainer, filepath)
        modelpath = filepath[:-4] + "pt"
        if trainer.is_global_zero:
            if os.path.exists(modelpath):
                os.remove(modelpath)


class LogAllLoss(pl.Callback):

    def __init__(self, properties) -> None:
        super().__init__()
        self.properties = properties
        self.train_loss = {p: [] for p in properties}
        self.train_loss['total'] = []
        self.title = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_rank == 0:
            loss_metrics = trainer.callback_metrics
            self.train_loss['total'].append(
                np.sqrt(loss_metrics['train_loss'].detach().cpu().numpy())
            )
            for prop in self.properties:
                self.train_loss[prop].append(
                    np.sqrt(loss_metrics[f'train_{prop}'].detach().cpu().numpy())
                )

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            if not self.title:
                content = f"{'epoch':^10}|{'step':^10}|{'lr':^10}|{'total':^21}"
                for prop in self.properties:
                    content += f"|{prop:^21}"
                log.info(content)
                self.title = True
            epoch = trainer.current_epoch
            step = trainer.global_step
            lr = trainer.optimizers[0].param_groups[0]["lr"]
            loss_metrics = trainer.callback_metrics
            train_loss = np.mean(self.train_loss['total'])
            val_loss = np.sqrt(loss_metrics['val_loss'].detach().cpu().numpy())
            content = f"{epoch:^10}|{step:^10}|{lr:^10.2e}|{train_loss:^10.4f}/{val_loss:^10.4f}"
            for prop in self.properties:
                train_prop_loss = np.mean(self.train_loss[prop])
                val_prop_loss = np.sqrt(
                    loss_metrics[f'val_{prop}'].detach().cpu().numpy()
                )
                content += f"|{train_prop_loss:^10.4f}/{val_prop_loss:^10.4f}"
            log.info(content)
            for prop in self.train_loss:
                self.train_loss[prop] = []


def update_dict(d1, d2):
    for key in d2:
        if key in d1 and isinstance(d1[key], dict):
            update_dict(d1[key], d2[key])
        else:
            d1[key] = d2[key]
    return d1


def get_stats(data_dict, dataset):
    if data_dict["meta"] is not None:
        with open(data_dict["meta"]) as f:
            stats = yaml.load(f, Loader=yaml.FullLoader)
        n_neighbor = stats["nNeighbor"]
        elements = stats["elements"]
        ground_energy = stats["ground_energy"]
        std = stats["std"]
        mean = stats["mean"]
    else:
        if type(data_dict["nNeighbor"]) is float:
            n_neighbor = data_dict["nNeighbor"]
        else:
            n_neighbor = float(dataset.n_neighbor_mean)

        if isinstance(data_dict["elements"], list):
            elements = data_dict["elements"]
        else:
            elements = dataset.all_elements

        if type(data_dict["groundEnergy"]) is float:
            ground_energy = [data_dict["groundEnergy"]] * len(elements)
        else:
            try:
                ground_energy = dataset.ground_energy
            except:
                ground_energy = [0.0] * len(elements)
        if type(data_dict["mean"]) is float:
            mean = data_dict["mean"]
        else:
            mean = 0.0
        if type(data_dict["std"]) is float:
            std = data_dict["std"]
        else:
            try:
                std = dataset.forces_std
            except:
                std = 1.0

    EnvPara.ELEMENTS = elements
    log.info(f"n_neighbor   : {n_neighbor}")
    log.info(f"all_elements : {elements}")
    log.info(f"ground_energy  : {ground_energy}")
    log.info(f"std   : {std}")
    log.info(f"mean  : {mean}")
    return ground_energy, mean, std, n_neighbor, elements


def get_cutoff(p_dict):
    cutoff = p_dict['cutoff']
    cut_dict = p_dict['Model']['CutoffLayer']
    if cut_dict['type'] == "cos":
        return CosineCutoff(cutoff=cutoff)
    elif cut_dict['type'] == "cos2":
        return SmoothCosineCutoff(cutoff=cutoff, cutoff_smooth=cut_dict['smoothCutoff'])
    elif cut_dict['type'] == "poly":
        return PolynomialCutoff(cutoff=cutoff, p=cut_dict['p'])
    else:
        raise Exception(
            "Unsupported cutoff type: {}, please choose from cos, cos2, and poly!".format(
                cut_dict['type']
            )
        )


def get_radial(p_dict, cutoff_fn):
    cutoff = p_dict['cutoff']
    radial_dict = p_dict['Model']['RadialLayer']
    if "bessel" in radial_dict['type']:
        radial_fn = BesselPoly(
            r_max=cutoff, n_max=radial_dict['nBasis'], cutoff_fn=cutoff_fn
        )
    elif "chebyshev" in radial_dict['type']:
        if "minDist" in radial_dict:
            r_min = radial_dict['minDist']
        else:
            r_min = 0.5
            log.warning(
                "You are using chebyshev poly as basis function, but does not given 'minDist', "
                "this may cause some problems!"
            )
        radial_fn = ChebyshevPoly(
            r_max=cutoff, r_min=r_min, n_max=radial_dict['nBasis'], cutoff_fn=cutoff_fn
        )
    else:
        raise Exception("Unsupported radial type: {}!".format(radial_dict['type']))
    if "MLP" in radial_dict['type']:
        if radial_dict["activateFn"] == "silu":
            activate_fn = nn.SiLU()
        elif radial_dict["activateFn"] == "relu":
            activate_fn = nn.ReLU()
        else:
            raise Exception(
                "Unsupported activate function in radial type: {}!".format(
                    radial_dict["activateFn"]
                )
            )
        return MLPPoly(
            n_hidden=radial_dict['nHidden'],
            radial_fn=radial_fn,
            activate_fn=activate_fn,
        )
    else:
        return radial_fn


def get_model(p_dict, elements, mean, ground_energy, std, n_neighbor):
    model_dict = p_dict['Model']
    target = p_dict['Train']['targetProp']
    target_way = {}
    if (
        ("energy" in target)
        or ("forces" in target)
        or ("virial" in target)
        or ("spin_torques" in target)
    ):
        target_way["site_energy"] = 0
    if "dipole" in target:
        target_way["dipole"] = 1
    if "polarizability" in target:
        target_way["polar_diag"] = 0
        target_way["polar_off_diagonal"] = 2
    if "direct_forces" in target:
        target_way["direct_forces"] = 1
    if "peratom_tensor" in target:
        target_way["peratom_tensor_diag"] = 0
        target_way["peratom_tensor_offdiag"] = 2
    if "l3_tensor" in target:
        target_way["l3_tensor_diag"] = 1
        target_way["l3_tensor_offdiag"] = 3
    if "peratom_l3_tensor" in target:
        target_way["peratom_l3_tensor_diag"] = 1
        target_way["peratom_l3_tensor_offdiag"] = 3
    cut_fn = get_cutoff(p_dict)
    emb = AtomicEmbedding(
        elements, model_dict['nEmbedding']
    )  # only support atomic embedding now
    radial_fn = get_radial(p_dict, cut_fn)
    max_r_way = expand_para(model_dict['maxRWay'], model_dict['nLayer'])
    max_out_way = expand_para(model_dict['maxOutWay'], model_dict['nLayer'])
    max_out_way[-1] = max(target_way.values())
    output_dim = expand_para(model_dict['nHidden'], model_dict['nLayer'])
    max_n_body = expand_para(model_dict['maxNBody'], model_dict['nLayer'])

    if model_dict['net'] == 'miao':
        model = MiaoNet(
            embedding_layer=emb,
            radial_fn=radial_fn,
            n_layers=model_dict['nLayer'],
            max_r_way=max_r_way,
            max_out_way=max_out_way,
            output_dim=output_dim,
            activate_fn=model_dict['activateFn'],
            target_way=target_way,
            mean=mean,
            std=std,
            norm_factor=n_neighbor,
        ).to(p_dict['device'])
    elif model_dict['net'] == 'miaomiao':
        model = MiaoMiaoNet(
            embedding_layer=emb,
            radial_fn=radial_fn,
            n_layers=model_dict['nLayer'],
            max_r_way=max_r_way,
            max_out_way=max_out_way,
            max_n_body=max_n_body,
            output_dim=output_dim,
            activate_fn=model_dict['activateFn'],
            target_way=target_way,
            mean=mean,
            std=std,
            norm_factor=n_neighbor,
            update_edge=model_dict['updateEdge'],
        ).to(p_dict['device'])
    elif model_dict['net'] == 'spinmiao':
        max_r_way = expand_para(
            model_dict['maxRWay'], model_dict['nLayer'] + model_dict['nSpinLayer']
        )
        max_out_way = expand_para(
            model_dict['maxOutWay'], model_dict['nLayer'] + model_dict['nSpinLayer']
        )
        output_dim = expand_para(
            model_dict['nHidden'], model_dict['nLayer'] + model_dict['nSpinLayer']
        )
        max_m_way = expand_para(model_dict['maxMWay'], model_dict['nSpinLayer'])
        spin_radial_fn = SpinChebyshevPoly(spin_max=p_dict['maxSpin'], n_max=12)
        model = SpinMiaoNet(
            embedding_layer=emb,
            spin_radial_fn=spin_radial_fn,
            radial_fn=radial_fn,
            n_layers=model_dict['nLayer'],
            n_spin_layers=model_dict['nSpinLayer'],
            max_r_way=max_r_way,
            max_m_way=max_m_way,
            soc=model_dict['soc'],
            max_out_way=max_out_way,
            output_dim=output_dim,
            activate_fn=model_dict['activateFn'],
            target_way=target_way,
            mean=mean,
            std=std,
            norm_factor=n_neighbor,
        ).to(p_dict['device'])

    assert isinstance(model_dict['Repulsion'], int), "Repulsion should be int!"
    module_dict = {'main': model}
    if "site_energy" in target_way:
        module_dict['ground_energy'] = GroundEnergy(
            atomic_number=elements, ground_energy=ground_energy
        )
    if model_dict['Repulsion'] > 0:
        assert (
            "site_energy" in target_way
        ), "Only support 'repulsion' when learning energy"
        module_dict['repulsion'] = TwoBody(
            embedding_layer=emb, cutoff_fn=cut_fn, k_max=model_dict['Repulsion']
        )

    return MultiAtomicModule(module_dict)


def main(
    *args, input_file='input.yaml', output_folder='outDir', load_model=None, load_checkpoint=None, **kwargs
):
    # Default values
    p_dict = DefaultPara
    with open(input_file) as f:
        update_dict(p_dict, yaml.load(f, Loader=yaml.FullLoader))
    p_dict["outputDir"] = os.path.join(p_dict["workDir"], output_folder)
    if os.path.exists(p_dict["outputDir"]):
        i = 1
        while os.path.exists(f"{p_dict['outputDir']}{i}"):
            i += 1
        shutil.move(p_dict["outputDir"], f"{p_dict['outputDir']}{i}")
        os.system(f"cp log.txt input.yaml allpara.yaml {p_dict['outputDir']}{i}")
    os.makedirs(p_dict["outputDir"])

    with open("allpara.yaml", "w") as f:
        yaml.dump(p_dict, f)

    setup_seed(p_dict["seed"])
    log.info("Using seed {}".format(p_dict["seed"]))

    EnvPara.ELEMENT_MODE = p_dict["Model"]["eleMode"]
    EnvPara.CONV_MODE = p_dict["Model"]["convMode"]
    EnvPara.EDGE_UPDATE_MODE = p_dict["Model"]["updateEdge"]
    if p_dict["precision"] == "double":
        EnvPara.FLOAT_PRECISION = torch.float64

    log.info(f"Preparing data...")
    dataset = LitAtomsDataset(p_dict)
    dataset.setup()
    ground_energy, mean, std, n_neighbor, elements = get_stats(p_dict["Data"], dataset)
    stats = {
        "cutoff": float(p_dict["cutoff"]),
        "nNeighbor": float(n_neighbor),
        "elements": elements,
        "ground_energy": (
            ground_energy.tolist()
            if isinstance(ground_energy, np.ndarray)
            else ground_energy
        ),
        "std": float(std),
        "mean": float(mean),
    }
    with open("metadata.yaml", "w") as f:
        yaml.dump(stats, f)

    if load_model is not None and 'ckpt' not in load_model:
        log.info(f"Load model from {load_model}")
        model = torch.load(load_model)
    else:
        ## TODO
        # mean and ground energy?
        model = get_model(p_dict, elements, mean, ground_energy, std, n_neighbor)
        model.register_buffer('all_elements', torch.tensor(elements, dtype=torch.long))
        model.register_buffer(
            'cutoff', torch.tensor(p_dict["cutoff"], dtype=torch.float64)
        )

    if load_model is not None and 'ckpt' in load_model:
        lit_model = LitAtomicModule.load_from_checkpoint(
            load_model, model=model, p_dict=p_dict
        )
    else:
        lit_model = LitAtomicModule(model=model, p_dict=p_dict)

    if load_checkpoint is not None:
        ckpt = torch.load(load_checkpoint)
        p_dict["Train"]["maxEpoch"] += ckpt['epoch']
        p_dict["Train"]["maxStep"] += ckpt['global_step']

    logger = pl.loggers.TensorBoardLogger(save_dir=p_dict["outputDir"])
    callbacks = [
        SaveModelCheckpoint(
            dirpath=p_dict["outputDir"],
            filename='{epoch}-{step}-{val_loss:.4f}',
            save_top_k=5,
            monitor="val_loss",
        ),
        LearningRateMonitor(),
        LogAllLoss(p_dict["Train"]['targetProp']),
    ]

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        default_root_dir='.',
        max_epochs=p_dict["Train"]["maxEpoch"],
        max_steps=p_dict["Train"]["maxStep"],
        enable_progress_bar=False,
        log_every_n_steps=p_dict["Train"]["logInterval"],
        val_check_interval=p_dict["Train"]["evalStepInterval"],
        check_val_every_n_epoch=p_dict["Train"]["evalEpochInterval"],
        gradient_clip_val=p_dict["Train"]["gradClip"],
    )

    if load_checkpoint is not None:
        log.info(f"Load checkpoints from {load_checkpoint}")
        trainer.fit(lit_model, datamodule=dataset, ckpt_path=load_checkpoint)
    else:
        trainer.fit(lit_model, datamodule=dataset)


if __name__ == "__main__":
    main()
