# Results

|     Model     | Min Validation Loss | Max Validation Accuracy | Min KL Div | Hidden Layers | Hidden Size | Epochs | Criterion | GPUs | Process Graph | Data Offset |                              wandb                              |
|:-------------:|:-------------------:|:-----------------------:|:----------:|:-------------:|:-----------:|:------:|:---------:|:----:|:-------------:|:-----------:|:---------------------------------------------------------------:|
|      MLP      |       0.03407       |          0.845          |            |       2       |     32      |   50   |    MSE    |  1   |       -       |      1      | [Link](https://wandb.ai/eriknikulski/carQUEST-ML/runs/keb7dita) |
|      MLP      |       0.0339        |          0.839          |            |       3       |     32      |   50   |    MSE    |  1   |       -       |      1      | [Link](https://wandb.ai/eriknikulski/carQUEST-ML/runs/yhdyz8mg) |
|      MLP      |       0.05391       |         0.6704          |            |       2       |     32      |   50   |    MSE    |  1   |       -       |      2      | [Link](https://wandb.ai/eriknikulski/carQUEST-ML/runs/c6j0tsrm) |
|      MLP      |       0.05354       |         0.6724          |            |       3       |     32      |   50   |    MSE    |  1   |       -       |      2      | [Link](https://wandb.ai/eriknikulski/carQUEST-ML/runs/9odr3re0) |
|      MLP      |       0.03759       |          0.292          |  0.03759   |       2       |     32      |   50   |   KLDiv   |  1   |       -       |      1      | [Link](https://wandb.ai/eriknikulski/carQUEST-ML/runs/r17tc0k5) |
| Embedding MLP |        0.164        |          0.25           |   0.1445   |       2       |     32      |   50   |    MSE    |  1   |       -       |      1      | [Link](https://wandb.ai/eriknikulski/carQUEST-ML/runs/tecj2bvw) |
|      MLP      |       0.2349        |         0.0255          |   0.333    |       2       |     32      |   50   |    MSE    |  1   |      <>       |      1      | [Link](https://wandb.ai/eriknikulski/carQUEST-ML/runs/675gjz93) |
|      MLP      |        0.235        |         0.0255          |   0.3332   |       3       |     32      |   50   |    MSE    |  1   |      <>       |      1      | [Link](https://wandb.ai/eriknikulski/carQUEST-ML/runs/hwhzb5zz) |



## Process State Autoencoder

https://wandb.ai/eriknikulski/carQUEST-ML/runs/1jept3vh (<>, MLP, 99.95% accuracy)

https://wandb.ai/eriknikulski/carQUEST-ML/runs/95t5rv7j (<>, MLP, 100% accuracy)


## System State Autoencoder

https://wandb.ai/eriknikulski/carQUEST-ML/runs/ai4qm3xv (<>, MLP, 100% accuracy) based on https://wandb.ai/eriknikulski/carQUEST-ML/runs/95t5rv7j


## System State Encoder-Decoder Offset 1

https://wandb.ai/eriknikulski/carQUEST-ML/runs/m2rayhyp (<>, MLP, 2.55% accuracy) based on https://wandb.ai/eriknikulski/carQUEST-ML/runs/ai4qm3xv