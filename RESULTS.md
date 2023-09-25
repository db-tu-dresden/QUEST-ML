# Results

|     Model     | Min Validation Loss | Max Validation Accuracy | Min KL Div | Hidden Layers | Hidden Size | Epochs | Criterion | GPUs | Process Graph | Data Offset |                              wandb                              |        Note        |
|:-------------:|:-------------------:|:-----------------------:|:----------:|:-------------:|:-----------:|:------:|:---------:|:----:|:-------------:|:-----------:|:---------------------------------------------------------------:|:------------------:|
|      MLP      |       0.03407       |          0.845          |            |       2       |     32      |   50   |    MSE    |  1   |       -       |      1      | [Link](https://wandb.ai/eriknikulski/carQUEST-ML/runs/keb7dita) |                    |
|      MLP      |       0.0339        |          0.839          |            |       3       |     32      |   50   |    MSE    |  1   |       -       |      1      | [Link](https://wandb.ai/eriknikulski/carQUEST-ML/runs/yhdyz8mg) |                    |
|      MLP      |       0.05391       |         0.6704          |            |       2       |     32      |   50   |    MSE    |  1   |       -       |      2      | [Link](https://wandb.ai/eriknikulski/carQUEST-ML/runs/c6j0tsrm) |                    |
|      MLP      |       0.05354       |         0.6724          |            |       3       |     32      |   50   |    MSE    |  1   |       -       |      2      | [Link](https://wandb.ai/eriknikulski/carQUEST-ML/runs/9odr3re0) |                    |
|      MLP      |       0.03759       |          0.292          |  0.03759   |       2       |     32      |   50   |   KLDiv   |  1   |       -       |      1      | [Link](https://wandb.ai/eriknikulski/carQUEST-ML/runs/r17tc0k5) |                    |
| Embedding MLP |        0.164        |          0.25           |   0.1445   |       2       |     32      |   50   |    MSE    |  1   |       -       |      1      | [Link](https://wandb.ai/eriknikulski/carQUEST-ML/runs/tecj2bvw) |                    |
|      MLP      |       0.2349        |         0.0255          |   0.333    |       2       |     32      |   50   |    MSE    |  1   |      <>       |      1      | [Link](https://wandb.ai/eriknikulski/carQUEST-ML/runs/675gjz93) | without reshaping  |
|      MLP      |        0.235        |         0.0255          |   0.3332   |       3       |     32      |   50   |    MSE    |  1   |      <>       |      1      | [Link](https://wandb.ai/eriknikulski/carQUEST-ML/runs/hwhzb5zz) | without reshaping  |
|      MLP      |       0.05008       |         0.5804          |  0.08038   |       3       |    2048     |   50   |    MSE    |  1   |      <>       |      1      | [Link](https://wandb.ai/eriknikulski/carQUEST-ML/runs/jku2i3lb) |                    |
|      MLP      |       0.02294       |         0.8115          |  0.03523   |       3       |    2048     |   50   |    MSE    |  1   |      <>       |      1      | [Link](https://wandb.ai/eriknikulski/carQUEST-ML/runs/to6ivw8w) | scaling_factor = 1 |
|      MLP      |       0.08307       |         0.3066          |   0.1404   |       5       |    1024     |   50   |    MSE    |  1   |      <>       |      1      | [Link](https://wandb.ai/eriknikulski/carQUEST-ML/runs/5xfzw710) |                    |
|      MLP      |       0.04838       |         0.5898          |  0.07867   |       5       |    1024     |  200   |    MSE    |  1   |      <>       |      1      | [Link](https://wandb.ai/eriknikulski/carQUEST-ML/runs/zfdirf83) |                    |



## Process State Autoencoder

https://wandb.ai/eriknikulski/carQUEST-ML/runs/1jept3vh (<>, MLP, 99.95% accuracy)

https://wandb.ai/eriknikulski/carQUEST-ML/runs/95t5rv7j (<>, MLP, 100% accuracy)


## System State Autoencoder

https://wandb.ai/eriknikulski/carQUEST-ML/runs/ai4qm3xv (<>, MLP, 100% accuracy) based on https://wandb.ai/eriknikulski/carQUEST-ML/runs/95t5rv7j


## System State Encoder-Decoder Offset 1

without reshaping: https://wandb.ai/eriknikulski/carQUEST-ML/runs/m2rayhyp (<>, MLP, 2.55% accuracy) based on https://wandb.ai/eriknikulski/carQUEST-ML/runs/ai4qm3xv


# Raytune

# <>
MLP without reshaping: https://wandb.ai/eriknikulski/carQUEST-ML/groups/ray-tune-84utx1bp (best accuracy 2.57%)
MLP: https://wandb.ai/eriknikulski/carQUEST-ML/groups/ray-tune-p4urg11i (best accuracy 57.01%)
MLP: https://wandb.ai/eriknikulski/carQUEST-ML/groups/ray-tune-0wg0ogg5 (best accuracy 55.25%)
MLP: https://wandb.ai/eriknikulski/carQUEST-ML/groups/ray-tune-l9p7icvk (best accuracy 57.87%)

Autoencoder MLP (PSE): https://wandb.ai/eriknikulski/carQUEST-ML/runs/5uupf9fw (98.9%)
Autoencoder MLP (SSE): https://wandb.ai/eriknikulski/carQUEST-ML/runs/cplc3upo (99.95%)
Offset 1 E-D MLP (SSE): https://wandb.ai/eriknikulski/carQUEST-ML/runs/nznfxoz1 (2.55%)


# System Model
### Process Autoencoder
https://wandb.ai/eriknikulski/carQUEST-ML/runs/0k6x5ht0
scaling_factor: 10
offset: 0
accuracy: 100%
converged after 1 epoch

### System Autoencoder (loaded and frozen Process Autoencoder; process decoder not frozen!; transformation model frozen)
https://wandb.ai/eriknikulski/carQUEST-ML/runs/qjnrag9a
scaling_factor: 10
offset: 0
accuracy: 76%
converged after 10 epochs

### System Autoencoder (loaded and frozen Process Autoencoder; process decoder not frozen!; transformation model frozen)
https://wandb.ai/eriknikulski/carQUEST-ML/runs/d124h6tj
scaling_factor: 1
offset: 0
accuracy: 73.5%
converged after 2 epochs

### System Autoencoder (loaded and frozen Process Autoencoder; transformation model not frozen!)
https://wandb.ai/eriknikulski/carQUEST-ML/runs/vosqaes0
scaling_factor: 10
offset: 0
accuracy: 33.55%
converged after 1 epoch

### System Model (Encoder and Decoder loaded; Encoder and Decoder frozen)
https://wandb.ai/eriknikulski/carQUEST-ML/runs/7ad5lys4
scaling_factor: 10
offset: 1
accuracy: 0.0%
converged after - epochs


https://wandb.ai/eriknikulski/carQUEST-ML/runs/vajnesxx
scaling_factor: 10
offset: 1
accuracy: 76.2%
converged after 20 epochs
Graph: -

## LSTM in decoder
#### Process Autoencoding
https://wandb.ai/eriknikulski/carQUEST-ML/runs/nlaidwsn

#### System Autoencoding
https://wandb.ai/eriknikulski/carQUEST-ML/groups/ray-tune-6jzlu6t2
max acc: 74.4 %

##### with Dropout
https://wandb.ai/eriknikulski/carQUEST-ML/groups/ray-tune-ge79l8tp

Dropout only in System Encoder/Decoder:
https://wandb.ai/eriknikulski/carQUEST-ML/groups/ray-tune-gfyk1f25

#### Only System with Offset 1
https://wandb.ai/eriknikulski/carQUEST-ML/groups/ray-tune-7g3mbth6
max acc: 50%

## Complete Model
https://wandb.ai/eriknikulski/carQUEST-ML/groups/ray-tune-aaxumfo2
offset: 1
max acc: 49.75%

https://wandb.ai/eriknikulski/carQUEST-ML/runs/l7r0r5z1

# System Encoder-Decoder

https://wandb.ai/eriknikulski/carQUEST-ML/groups/ray-tune-6xzym8ai
max acc: 65.5%

https://wandb.ai/eriknikulski/carQUEST-ML/runs/08nc7nss
max acc: 74.18%

https://wandb.ai/eriknikulski/carQUEST-ML/runs/46thzggd
only_system: True
offset: 1
max acc: 44.9%

https://wandb.ai/eriknikulski/carQUEST-ML/runs/b1ryqi9a
offset: 1
max acc: 11.25%

https://wandb.ai/eriknikulski/carQUEST-ML/runs/m3k3rz69
<>
only_system: True
offset: 1
max acc: 14.9%

https://wandb.ai/eriknikulski/carQUEST-ML/runs/zpnh7z28
<>
offset: 1
max acc: 2.35%


## FLAT MLP
https://wandb.ai/eriknikulski/carQUEST-ML/runs/97i50ek5
hidden size: 2048
hidden layers: 3

offset 1:
    100% acc nach 2 epochs


### clpx graph
https://wandb.ai/eriknikulski/carQUEST-ML/runs/y970uz4x
~60%

https://wandb.ai/eriknikulski/carQUEST-ML/runs/e85j1yc4
~65%

Tuning lambda:

Best trial config: {'lambda': 0.0, 'hidden_size': 2048, 'hidden_layers': 3, 'learning_rate': 0.012786501847370822}
Best trial final validation loss: 0.06274252384901047
Best trial final validation accuracy: 0.02552083507180214

Best trial config: {'lambda': 0.2, 'hidden_size': 1024, 'hidden_layers': 2, 'learning_rate': 0.006729034236445496}
Best trial final validation loss: 0.05706259235739708
Best trial final validation accuracy: 0.5041667222976685

Best trial config: {'lambda': 0.4, 'hidden_size': 1024, 'hidden_layers': 2, 'learning_rate': 0.01554257253318455}
Best trial final validation loss: 0.0453731007874012
Best trial final validation accuracy: 0.6171875596046448

Best trial config: {'lambda': 0.45, 'hidden_size': 512, 'hidden_layers': 2, 'learning_rate': 0.049408163272648124}
Best trial final validation loss: 0.053164027631282806
Best trial final validation accuracy: 0.49427086114883423

Best trial config: {'lambda': 0.5, 'hidden_size': 1024, 'hidden_layers': 2, 'learning_rate': 0.032189452243549674}
Best trial final validation loss: 0.04268467798829079
Best trial final validation accuracy: 0.6401041746139526

Best trial config: {'lambda': 0.6, 'hidden_size': 512, 'hidden_layers': 2, 'learning_rate': 0.04407393248372669}
Best trial final validation loss: 0.0770404115319252
Best trial final validation accuracy: 0.3151041865348816

Best trial config: {'lambda': 0.8, 'hidden_size': 512, 'hidden_layers': 2, 'learning_rate': 0.015696266176867577}
Best trial final validation loss: 0.04444761574268341
Best trial final validation accuracy: 0.3541666865348816

Best trial config: {'lambda': 1.0, 'hidden_size': 512, 'hidden_layers': 2, 'learning_rate': 0.009709370093513872}
Best trial final validation loss: 0.04658373445272446
Best trial final validation accuracy: 0.29739585518836975


https://wandb.ai/eriknikulski/carQUEST-ML/runs/x3yvh28r 65.78%; Acc. Window 100

https://wandb.ai/eriknikulski/carQUEST-ML/runs/inolehud 70.31%; Acc. Window 100; 5 Prev. States 
