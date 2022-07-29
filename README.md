# Self Explainable Transformers for Flow Cytometry Cell Classification

Official implementation of our work: *Towards Self-Explainable Transformers for Cell Classification in Flow Cytometry Data* by Florian Kowarsch, Lisa Weijler, Matthias Wödlinger, Michael Reiter, Margarita Maurer-Granofszky, Angela Schumich, Elisa O. Sajaroff, Stefanie Groeneveld-Krentz, Jorge G. Rossi, Leonid Karawajew, Richard Ratei, Michael N. Dworzak

## Installation

All decencies are provided in *requirements.txt*. Install with:
```
pip install -r requirements.txt
```

## Usage

In order to perform the experiments from the paper the following steps must be followed:
1. Create preprocessed cache files from FCM files.
2. Train model with the created cache files.
3. Test a trained model


### Creating Cache

 [createcache.py](createcache.py) generates one cache file per FCM-sample which includes the necessary data to training the model. Preprocessing steps such as computing the convex hull for the specified Gate-Definition is applied as well as determining the ground truth class for every cell.

 ```json
 {
    "type_name": "src.datastructures.configs.cachedatacreationconfig.CacheDataCreationConfig",
    "output_location": "path to folder where cached files should be stored",
    "blacklist_path": "",   //optional text file that specfies FCM-files that should be skipped
    "ignore_blacklist": true,
    "outlier_handler_config": {
        "n_events_threshold": 300, // min number of events needed bevore outlier removal is executed
        "alpha": 0.00001 // alpha value for Mahalanobis outlier removal
    },
    "source_datasets": [] // list of datasets that should be used
    "gate_defintions": [] // definition of gates from which the convex hull should be created
}
 ```

 ### Train Model

[train.py](train.py) serves as entrypoint for model training.

```json
{
    "type_name": "src.datastructures.configs.trainconfig.TrainConfig",
    "name": "train_vie14_val_bln",
    "default_retrieve_options": {
        "shuffle": true,
        "use_convex_gates": true, //wheter actual human gt polygons or generated convex gates are used
        "filter_gate" : "Intact", //Gate after which events are considered in training
        "polygon_min": -0.1,
        "polygon_max": 1.7,
        "always_keep_blasts": true, //wheter blast should be favored when sampling events
        "gate_polygon_interpolation_length" : 120, // number of points per polygon that are interpolated
        "gate_polygon_seq_length": 20, //number of points per polygon
        "events_seq_length": 50000, //number of events per sample used for training
        "used_markers": [],      // names of used markers
        "used_gates": [],        // names of used gates
        "gate_definitions" : [], //used Gate Definitions
        "events_mean": [
            1.2207406759262085,
            1.245536208152771,
            1.413953185081482,
            0.9958911538124084,
            2.1471059322357178,
            2.0955066680908203,
            1.4734785556793213,
            0.42288827896118164,
            0.8758889436721802,
            2.472586154937744
        ],
        "events_sd": [
            0.3791336715221405,
            0.3249945342540741,
            0.3320983946323395,
            0.18722516298294067,
            0.35046276450157166,
            0.25932908058166504,
            0.4003131091594696,
            0.014336027204990387,
            0.5279378294944763,
            0.34515222907066345
        ],
        "augmentation_config": {
            "shift_propability": 0.7,
            "shift_percent": 0.25,
            "polygon_scale_range" : {
                "Syto" : 0.01,
                "Singlets" : 0.01,
                "Intact" : 0.05,
                "CD19" : 0.15,
                "Blasts_CD45CD10" : 0.3,
                "Blasts_CD20CD10" : 0.3,
                "Blasts_CD38CD10" : 0.3
            },
            "scale_propability" : 0.7,
            "scale_propability_2nd_marker" : 0.3
        }
    },
    "train_data":  {}, //dataset for training
    "validation_data": {}, //dataset for validation
    "model_storage": {
        "file_path": "./data/saved_models/train_vie14_val_bln",
        "load_stats_from_file": false,
        "gpu_name" :"cuda"
    },
    "train_params": {
        "learning_rate": 0.001,
        "weight_decay": 0.00000000000001,
        "validation_interval": 50,
        "n_training_epochs": 1500,
        "training_batchsize": 2,
        "clip_norm": 4.0,
        "random_seed": 42,
        "polygon_loss_weight": 1.0,
        "saving_interval": 50,
        "use_auxiliary_loss" : true
    },
    "model_factory": {
        "model_type": "src.model.FlowGATR.FlowGATR",
        "params_type": "src.datastructures.configs.modelparams.ModelParams",
        "params": {
            "dim_input": 10,
            "n_hidden_layers_ISAB": 2,
            "n_hidden_layers_decoder": 2,
            "n_obj_queries": 7,
            "points_per_query" : 5,
            "dim_latent": 36,
            "n_polygon_out": 20,
            "n_decoder_cross_att_heads": 6,
            "n_hidden_layers_polygon_out": 2,
            "n_perciever_blocks_decoder" : 4
        }
    },
    "wandb_config": {
        "entity": "your_wandb_username",
        "prj_name": "fcm-polygon-pred",
        "notes": "",
        "tags": [],
        "enabled": true //wheter training is logged to wandb or not
    },
    "gpu_name": "cuda",
    "n_workers": 0
}
```

 ### Test Model

 [test.py](test.py) allows to evaluate the performance of an already trained model on a given test set.

 ## Data

 ## Cite