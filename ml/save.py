import torch

import const
import ddp

BEST_LOSS = float('inf')
LAST_EPOCH = 0
CURRENT_SAVE_ID = 0


def model(*args, **kwargs):
    if ddp.is_main_process():
        torch.save(*args, **kwargs)


def is_ready_to_save(loss, epoch):
    global BEST_LOSS, LAST_EPOCH
    epoch_dist = epoch - LAST_EPOCH
    if loss < BEST_LOSS and epoch >= const.MIN_CHECKPOINT_EPOCH and epoch_dist >= const.MIN_CHECKPOINT_EPOCH_DIST:
        BEST_LOSS = loss
        LAST_EPOCH = epoch
        return True
    return False


def checkpoint_encoders_decoders(epoch, joint_translator, loss, base_path):
    if ddp.is_main_process():
        if is_ready_to_save(loss, epoch):
            joint_module = getattr(joint_translator, 'module', joint_translator)
            for i, encoder in enumerate(joint_module.encoders):
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': encoder.state_dict(),
                    'loss': loss,
                }, base_path + f'_encoder_{i}_{epoch}.pt')
                print(f'saved encoder_{i} checkpoint at epoch: {epoch} with loss: {loss}')

            for i, decoder in enumerate(joint_module.decoders):
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': decoder.state_dict(),
                    'loss': loss,
                }, base_path + f'_decoder_{i}_{epoch}.pt')
                print(f'saved decoder checkpoint at epoch: {epoch} with loss: {loss}')
